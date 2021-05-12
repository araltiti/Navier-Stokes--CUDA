#include "velocityBC.cuh"

__global__ void velocityBC(float* u, float* v, unsigned int N, float Ulid){

    int i = threadIdx.x, j = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, dim = blockDim.x; 
    int col = bx*dim + i;
    int row = by*dim + j;
    int index = row*(N+1) + col;
    int indexV = row*N + col;


    if(index >= N*(N+1) || indexV >= N*(N+1))
        return;
    

    if(index%(N+1) == 0){
        u[index] = -u[index+1];    
    }
    if(index%(N+1) == N){
        u[index] = 2*Ulid - u [index-1];
    }
    if(indexV/(N+1) == 0){
        v[indexV] = -v[indexV+N];
    }
    if(indexV/(N+1) == N){
        v[indexV] = -v[indexV - N];
    }
}