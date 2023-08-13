/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
#define BlockSize 1024
 
using namespace std;



ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/
// Initialize kernel with graph data structures and parameters
__global__ void ini_kernel( int *active,int *num_active, int *levels, int V, int E, int L,int *cg,int *csr_offset, int *csr_List, int *apr, int *aid)
{
    // Calculate thread ID
    int a,b,c,d;
    a=threadIdx.x;
    b = blockIdx.x;
    c = blockDim.x;
    d = a + blockIdx.x * c;
    int tid = d;
    // If the thread is within the vertex range and the corresponding predecessor count is zero
    if(tid < V && apr[tid] == 0)
    {
            // Activate the vertex, set its level to 0, increment the number of active vertices, and update the cg (last active vertex)
            atomicExch(&active[d], 1);
            atomicExch(&levels[a + b * c], 0);
            atomicAdd(&num_active[ (d%1) ], 1); 
            atomicMax(&cg[0], tid);
        
    }
}

// Phase one: Activate vertices with a level equal to the current level and enough predecessors


__global__ void first(int *aid, int *num_active, int *active, int *levels, int V, int E, int L,int l,int *gmax, int *gmin,int *csr_offset, int *csr_List, int *apr)
 {
      
            // Calculate thread ID

        int tid = threadIdx.x + blockIdx.x * blockDim.x + gmin[0];
  
        int curr_level = l;
        // If the thread is outside the vertex range, return

        if(tid >= V)
        {return ; }
        
         // If the vertex is inactive and has a level equal to the current level   
        if(levels[tid] == l && active[tid]==0 )
        {
            // If the vertex has enough predecessors, activate the vertex and increment the number of active vertices at the current level

            if(aid[tid] >= apr[tid] && tid >= 0)
            {
                atomicExch(&active[(threadIdx.x + blockIdx.x * blockDim.x + gmin[0])], 1);
                atomicAdd(&num_active[curr_level], 1);
             }
        }
        __syncthreads();
        
 }
 // Phase two: Deactivate vertices surrounded by inactive vertices with the same level
__global__ void second(int *aid, int *num_active, int *active, int *levels, int V, int E, int L,int l,int *gmax, int *gmin, int *csr_offset, int *csr_List, int *apr)
    {   
        // Calculate thread ID 
        int tid = threadIdx.x + blockIdx.x * blockDim.x + gmin[0];
        int next = tid + 1;
        int prev = tid - 1;
        int curr_level = l;

        if(tid >= V || levels[tid] != curr_level || active[tid] == 0) {
            return ;
        }
        // If the thread is outside the vertex range or the vertex level is different from the current level or the vertex is inactive, return
        if(tid > 0 && tid < (V - 1) && active[tid-1] == 0 && active[next] == 0)
        {
            
                if(levels[prev] == l && levels[next] == curr_level) 
                {
                    atomicExch(&active[(threadIdx.x + blockIdx.x * blockDim.x + gmin[0])], 0);
                    atomicSub(&num_active[curr_level], 1);
                }
            
        }
        __syncthreads();
        
    }
// Phase three: Update the aid array and vertex levels based on active vertices and their neighbors
__global__ void third(int *aid, int *num_active, int *active, int *levels, int V, int E, int L,int l,int *gmax, int *gmin, int *csr_offset, int *csr_List, int *apr)
{    
        // Calculate thread ID
        int tid = threadIdx.x + blockIdx.x * blockDim.x + gmin[0];
        // If the thread is outside the vertex range or the vertex level is different from the current level, return
        if(tid >= V || levels[tid] != l)
        {
            return ;
        }
        // Iterate over the neighbors of the current vertex
        for(int j=csr_offset[tid] ; j<csr_offset[tid+1] ; j++)
            {
                // If the current vertex is active
                if(active[(threadIdx.x + blockIdx.x * blockDim.x + gmin[0])] == 1)
                {
                  // Increment the aid value of the neighboring vertex  
                  atomicAdd(&aid[csr_List[j]], 1);
                }
                // Update the level of the neighboring vertex and update the global maximum active vertex
                levels[csr_List[j]] = l+1;
                atomicMax(&gmax[0],csr_List[j]);
                
            }
        // Synchronize threads
       __syncthreads();
      
     
    
}
    
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/
// Set the grid size for kernel launches based on the number of vertices and block size
int grid_size = ceil((float)V / BlockSize) , cc[1];
int *d_levels;
cudaMalloc(&d_levels, V*sizeof(int));
cc[0] = 0;
// Initialize levels array on device
cudaMemset(d_levels, -1, V*sizeof(int));
int *cg;
cudaMemset(d_aid, 0, V*sizeof(int));



int cmax[1], *gmax,*gmin, cmin[1];

cmax[0] = 0;
// Allocate device memory for cg, gmax, and gmin
cudaMalloc(&cg, 1*sizeof(int));
cudaMalloc(&gmax, 1*sizeof(int));
cudaMalloc(&gmin, 1*sizeof(int));
// Initialize cg to 0
cudaMemset(cg, 0, 1 * sizeof(int));
int *d_active;
cudaMalloc(&d_active, V*sizeof(int));
cmin[0] = 0;
// Initialize active array on device
cudaMemset(d_active, 0, V*sizeof(int));
int block = 1024;
// Call the ini_kernel to initialize active vertices, levels, and cg
ini_kernel<<<grid_size,block>>>(d_active,d_activeVertex, d_levels,V,E,L,cg,d_offset,d_csrList,d_apr,d_aid);
cudaMemcpy(cc, cg, 1*sizeof(int), cudaMemcpyDeviceToHost);
cmax[0] = cc[0];
cudaMemcpy(gmax, cmax, sizeof(int), cudaMemcpyHostToDevice);
// Loop over all levels
for(int levej = 0; levej < L; levej++)
{
  int range = cmax[0] - cmin[0] + 1;
  int grid1 = ceil((float)(range)/1024);
  // Call the first, second, and third kernels to process each level
  first<<<grid1,BlockSize>>>(d_aid,d_activeVertex,d_active,d_levels,V,E,L,levej,gmax,gmin,d_offset,d_csrList,d_apr);
  second<<<grid1,1024>>>(d_aid,d_activeVertex,d_active,d_levels,V,E,L,levej,gmax,gmin,d_offset,d_csrList,d_apr);
  third<<<grid1,BlockSize>>>(d_aid,d_activeVertex,d_active,d_levels,V,E,L,levej,gmax,gmin,d_offset,d_csrList,d_apr);
   // Update cmin and cmax
  int max = cmax[0];
  cmin[0] = max;
  cudaMemcpy(cmax, gmax, sizeof(int), cudaMemcpyDeviceToHost);
    if(levej < 0)
    {
        break;
    }
  // Copy the updated cmin to the device
  cudaMemcpy(gmin, cmin, sizeof(int), cudaMemcpyHostToDevice);
  
  
}
cmin[0] = 0;
// Free the device memory for csr_List
cudaFree(d_csrList);
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);


cmax[0] = 0;
// Free the device memory for offset
cudaFree(d_offset);


/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
