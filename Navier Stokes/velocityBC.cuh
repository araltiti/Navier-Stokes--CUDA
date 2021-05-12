#ifndef VELOCITYBC_CUH
#define VELOCITYBC_CUH


//blockDim = 3x3
//griddim = max(N/2,N+1/2)  + blockDim -1 / blockDim
__global__ void velocityBC(float* u, float* v, unsigned int N, float Ulid);


#endif