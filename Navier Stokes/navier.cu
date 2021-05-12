#include "convection.cuh"
#include "gSource.cuh"
#include "pressure.cuh"
#include "velocityBC.cuh"
#include "corrector.cuh"
#include <iostream>
#include <algorithm>


#define BLOCK_DIM 8

int main(int agrc, char* argv[]){

    int nIterations = 500;
    unsigned int n = 21;
    float dx = 0.05;
    float dy = 0.05;
    float dt = 0.05;
    float nu = 0.01;
    float rho = 1;
    float tf = 4.5;
    float Ulid = 1;

    float* u = new float[n*(n+1)];  //N rows and N+1 cols
    float* v = new float[n*(n+1)];  //N+1 rows and N cols
    float* uStar = new float[n*(n+1)];
    float* vStar = new float[n*(n+1)];
    float* g = new float[n*n];
    float* p = new float[(n+1)*(n+1)];

  

    cudaMallocManaged((void **)&u, n*(n+1)*sizeof(float));
    cudaMallocManaged((void **)&v, n*(n+1)*sizeof(float));
    cudaMallocManaged((void **)&uStar, n*(n+1)*sizeof(float));
    cudaMallocManaged((void **)&vStar, n*(n+1)*sizeof(float));
    cudaMallocManaged((void **)&g, n*n*sizeof(float));
    cudaMallocManaged((void **)&p, (n+1)*(n+1)*sizeof(float));
    
    std::fill(u,u + n*(n+1), 0);
    std::fill(v, v+n*(n+1), 0);
    std::fill(uStar,uStar+n*(n+1), 0);
    std::fill(vStar,vStar+n*(n+1), 0);
    std::fill(g,g+n*n, 0);
    std::fill(p,p+(n+1)*(n+1), 0);

    //Initializing u-velocity
    for(unsigned int i = 0; i < n; i++)
        u[i * (n+1) + n] = 2;


    dim3 dimBlock(BLOCK_DIM,BLOCK_DIM);
    size_t grid_dim = (n + BLOCK_DIM - 1)/BLOCK_DIM;
    dim3 dimGrid(grid_dim, grid_dim);


    for(float time = 0; time < tf; time += dt){
        int Ns = (2*BLOCK_DIM*BLOCK_DIM) * sizeof(float);
        convection<<<dimGrid, dimBlock, Ns>>>(u,v,uStar,vStar,n,dx,dy,dt, nu);

               
        gSource<<<dimGrid, dimBlock>>>(g,uStar,vStar,n,rho,dx,dy,dt);


        for(int i = 0; i < nIterations; i++){
            pressure<<<dimGrid,dimBlock>>>(dx,p,g,n);

        }

        corrector<<<dimGrid,dimBlock>>>(uStar,vStar,u,v,p,n,dx,dy,dt);
   

        velocityBC<<<dimGrid,dimBlock>>>(u,v,n,Ulid);
        cudaDeviceSynchronize();

    }
    
    
    
    for (int i = 0; i<42;i++)
        std::cout << u[i] << "\n";


    return 0;

}