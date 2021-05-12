#include "convection.cuh"
#include <math.h>
#include <stdio.h>


__global__ void convection(float* u, float* v, float* uStar, float* vStar, unsigned int N1, float dx, float dy, float dt, float nu){

    int N = (int)N1;
    float d2udx2, d2udy2, du2dx, duvdy;
    float d2vdx2, d2vdy2, dv2dy, duvdx;
    int i = threadIdx.x, j = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, dim = blockDim.x; 
    int col = bx*dim + i;
    int row = by*dim + j;
    int index = row*(N+1) + col;
    int indexV = row*N + col;

    if (index >= N*(N+1)) 
        return;

    if(indexV >= N*(N+1))
        return;
    extern __shared__ float prev[];

    float* u_prev = prev;
    if (index < N*(N+1)){
        u_prev[j*dim + i] = u[index];
    }

    float* v_prev = &u_prev[dim*dim];
    if(indexV < N*(N+1)){
        v_prev[j*dim + i] = v[indexV];
    }
    __syncthreads();

    bool blockCheck = ((i>0) && (i<dim-1) && (j>0) && (j<dim-1));
    bool boundCheckU = ((index > N) && (index < N*(N+1)-N-1) && (index%(N+1) !=0) && (index%(N+1)!=N));
    bool boundCheckV = ((indexV > N-1) && (indexV < N*(N+1)-N) && (indexV%N != 0) && (indexV%N != N-1));


    //u velocity predictor

    if(blockCheck && boundCheckU && boundCheckV){
        d2udx2 = (u_prev[(j+1)*dim+i] - 2*u_prev[j*dim+i] + u_prev[(j-1)*dim + i])/(dx*dx);
        d2udy2 = (u_prev[j*dim + i+1] -2*u_prev[j*dim+i] + u_prev[j*dim + i - 1]) / (dy*dy);
        du2dx = 1/dx*( powf((u_prev[j*dim+i] + u_prev[(j+1)*dim + i])/2, 2) - powf((u_prev[(j-1)*dim + i]+  u_prev[j*dim + i])/2, 2));
        duvdy =  1/dy*(((v_prev[j*dim + i] + v_prev[(j+1)*dim + i])/2)*((u_prev[j*dim + i+1]+u_prev[j*dim + i])/2) 
            - ((v_prev[j*dim + i-1]+v_prev[(j+1)*dim + i-1])/2)*((u_prev[j*dim + i-1]+u_prev[j*dim + i])/2));

        uStar[index] = u_prev[i*dim +j]+ dt*(nu*(d2udx2+d2udy2) - du2dx - duvdy);
         __syncthreads();   
    } 

    else if(boundCheckU){
        d2udx2 = (u[index+N+1] - 2*u[index] + u[index-N-1])/(dx*dx);
        d2udy2 = (u[index+1] -2*u[index] + u[index - 1]) / (dy*dy);
        du2dx = 1/dx*( powf((u[index] + u[index+N+1])/2, 2) - powf((u[index-N-1]+  u[index])/2, 2));
        duvdy =  1/dy*((v[indexV] + v[indexV+N]/2)*((u[index + 1]+u[index])/2) - ((v[indexV - 1] 
            + v[indexV + N- 1])/2)*((u[index - 1]+u[index])/2));

        uStar[index] = u[index]+ dt*(nu*(d2udx2+d2udy2) - du2dx - duvdy);
        __syncthreads();
    }

    //v-velocity predictor

     if(blockCheck && boundCheckU && boundCheckV) {

        d2vdx2 = (v_prev[(j+1)*dim + i] - 2*v_prev[j*dim + i] + v_prev[(j-1)*dim+1])/(dx*dx);
        d2vdy2 = (v_prev[j*dim+i+1] - 2*v_prev[j*dim+i] + v_prev[j*dim+i-1])/(dy*dy);
        dv2dy = 1/dy*( powf((v_prev[j*dim+i] +v_prev[j*dim+i+1])/2,2) - powf((v_prev[j*dim+i-1] + v_prev[j*dim+i])/2,2));  
        duvdx = 1/dx*(((v_prev[(j+1)*dim+i] + v_prev[j*dim+i])/2) *((u_prev[j*dim+i+1]+u_prev[j*dim+i])/2) - 
            ((v_prev[(j-1)*dim+i] + v_prev[j*dim+i])/2) *((u_prev[(j-1)*dim+i+1] +u_prev[(j-1)*dim+i])/2));

        vStar[indexV] = v[indexV] + dt*(nu*(d2vdx2+d2vdy2) -dv2dy -duvdx);
        __syncthreads();
    } 

    else if(boundCheckV){
        d2vdx2 = (v[indexV+N] - 2*v[indexV] + v[indexV-N])/(dx*dx);
        d2vdy2 = (v[indexV+1] -2*v[indexV] + v[indexV - 1]) / (dy*dy);
        dv2dy = 1/dx*( powf((v[indexV] + v[indexV+1])/2, 2) - powf((v[indexV-1]+  v[indexV])/2, 2));
        duvdx =  1/dy*((v[indexV] + v[indexV+N]/2)*((u[index + 1]+u[index])/2) - ((v[indexV - N] 
            + v[indexV])/2)*((u[index - N - 1 +1]+u[index-N-1])/2));

        vStar[indexV] = v[indexV] + dt*(nu*(d2vdx2+d2vdy2) - dv2dy - duvdx);
        __syncthreads();
    }

}