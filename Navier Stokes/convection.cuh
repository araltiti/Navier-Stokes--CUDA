
#ifndef CONVECTION_CUH
#define CONVECTION_CUH


//blockDim = 3x3
//griddim = max(N/2,N+1/2)  + blockDim -1 / blockDim
__global__ void convection(float* u, float* v, float* uStar, float* vStar, unsigned int n, float dx, float dy, float dt, float nu);


#endif