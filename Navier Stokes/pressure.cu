#include "pressure.cuh"
#include <math.h>

#define PI_F 3.141592654f
__global__ void pressure(float dx, float* p, float* g, int N) {	

	int i = threadIdx.x, j = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, dim = blockDim.x;
    float e_e, e_w, e_n;

	int col = bx*dim + i;
    int row = by*dim + j;
    int indexG = row*N + col;
    int indexP = row*(N+1) + col;

    float lambda_jac = 0.5 * (cos(PI_F/N) + cos(PI_F/N));
    float omega = 2/(1 + sqrt(1 - lambda_jac*lambda_jac));

    if (indexP >= (N+1)*(N+1))
        return;


    if(indexP%(N+1) < N-1){
        e_n = 1;
    }else if(indexP%(N+1) == N-1){
        e_n = 0;
    }

    if(indexP/(N+1) < N-1){
        e_e=1;
    }else if(indexP/(N+1) == N-1){
        e_e =0;
    }

    if(indexP/(N+1) > 1){
        e_w = 1;
    }else if(indexP/(N+1) == 1){
        e_w =0;
    }

    bool boundCheckP = ( (indexP > N) && (indexP < N*(N+1)-N-1) && (indexP%(N+1) != 0) && (indexP%(N+1) != N));

	if (boundCheckP) {
		p[indexP] = p[indexP]*(1-omega)+ omega/(1+e_e + e_w + e_n)*(e_e*p[indexP + N + 1] + e_w*p[indexP - N-1]+
		e_n*p[indexP +1]+p[indexP - 1]-g[indexG]*(dx*dx));

		__syncthreads();

	}

}
 