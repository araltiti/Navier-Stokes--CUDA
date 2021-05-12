#ifndef CORRECTOR_CUH
#define CORRECTOR_CUH


//blockDim = 3x3
//griddim = max(N/2,N+1/2)  + blockDim -1 / blockDim
__global__ void corrector(float* uStar, float* vStar, float* uCorr, float* vCorr, float* P, int n, float dx, float dy, float dt);


#endif
