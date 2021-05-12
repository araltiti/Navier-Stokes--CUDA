#include "corrector.cuh"
#include <math.h>
#include <stdio.h>

__global__ void corrector(float* uStar, float* vStar, float* uCorr, float* vCorr, float* P, int N, float dx, float dy, float dt){

    int i = threadIdx.x, j = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, dim = blockDim.x; 
    int col = bx*dim + i;
    int row = by*dim + j;
    int index = row*(N+1) + col;
    int indexV = row*N + col;
    int indexP = row*(N+1) + col;

    if (index >= N*(N+1)) 
        return;

    bool boundCheckU = ((index > N) && (index < N*(N+1)-N-1) && (index%(N+1) !=0) && (index%(N+1)!=N));
    bool boundCheckV = ((indexV > N-1) && (indexV < N*(N+1)-N) && (indexV%N != 0) && (indexV%N != N-1));


    if(boundCheckU){
        uCorr[index] = uStar[index] - dt/dx*(P[indexP+N+1]-P[indexP]);
        __syncthreads();
    }

    if(boundCheckV){
        vCorr[indexV] = vStar[indexV] - dt/dy*(P[indexP+1]-P[indexP]);
        __syncthreads();
    }
}
