
#ifndef GSOURCE_CUH
#define GSOURCE_CUH


//blockDim = 3x3
//griddim = max(N/2,N+1/2)  + blockDim -1 / blockDim
__global__ void gSource(float* g, float* uStar, float* vStar, int N, float rho, float dx, float dy, float dt);


#endif