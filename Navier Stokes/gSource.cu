#include "gSource.cuh"


__global__ void gSource(float* g, float* uStar, float* vStar, int N, float rho, float dx, float dy, float dt){


    int i = threadIdx.x, j = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, dim = blockDim.x; 
    int col = bx*dim + i;
    int row = by*dim + j;
    int index = row*(N+1) + col;
    int indexV = row*N + col;
    int indexG = row*N + col;

    if (index >= N*(N+1)) 
        return;
    if(indexV >= N*(N+1))
        return;
    if(indexG >= N*N)
        return;
    
    bool boundCheckG = ( (indexG > N-1)  && (indexG%N != 0));

    if(boundCheckG){
        g[indexG] = rho/dt* ( (uStar[index] - uStar[index - N-1])/dx + (vStar[indexV] - vStar[indexV-1])/dy);
        __syncthreads();
    }
    
}