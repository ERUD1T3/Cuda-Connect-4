/* Assignment 1 source code */

// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "libs/cudmatrix.h"

//constants
#define K 1000 // number of matvec iterations
#define MAX_BLOCK_SIZE 256

const float Q = .15;
//////////////////////////////////////////////////////////////////////////////////////////////////////////

/*Non parallel */
void minmaxPageRank(Vector *vec);
void dampen(DMatrix *H); // transform H matrix into G (dampened) matrix
/* parallel */
__global__ void d_normalize(float *d_v, int rowSize, int colSize, float *sum);
__global__ void d_mult(float *vec, float *mat, float *out, const int colSize, const int rowSize);
void vecNormalize(Vector *vec);            // normalize values of surfer values
Vector *matVec(DMatrix *mat, Vector *vec); // multiply compatible matrix and vector

int main(int argc, char *argv[])
{
    //reading number of pages from terminal
    uint numpg = (argc > 1) ? atoi(argv[1]) : 16;

    printf("-------Dense Matrix Test-----------------------\n\n");
    // create the H matrix
    DMatrix *H = initDMatrix(numpg);

    // create and initialize at the pagerank Vector
    Vector *pgrkV = initVector(numpg);

    clock_t startTime, endTime;

    // display the H matrix
    // printDMatrix(H);

    //prints pagerank vector before matvec
    // printf("pagerank vector before web surfing\n");
    // printDMatrix(pgrkV);

    startTime = clock();

    dampen(H);

    // apply matvec with dampening on for 1000 iterations
    for (uint iter = 0; iter < K; ++iter)
    {
        pgrkV = matVec(H, pgrkV); // parallelized matVecDampn
        // printf("pagerank after iter %d\n", iter);
        // printDMatrix(pgrkV);
    }

    if (numpg <= 16)
    { // print the page rank vector is small
        printf("pagerank vector after web surfing\n");
        printDMatrix(pgrkV);
    }

    // display lowest and highest page ranks
    minmaxPageRank(pgrkV);

    endTime = clock();
    printf("\nruntime = %.16e\n", ((float)(endTime - startTime)) / CLOCKS_PER_SEC);

    // garbage management
    destroyDMatrix(H);
    destroyDMatrix(pgrkV);

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// definition of dense matrix object
void minmaxPageRank(Vector *vec)
{
    // return the max and min values in the vector, as well as their indices

    float minval = vec->data[0 * vec->colSize + 0],
           maxval = vec->data[0 * vec->colSize + 0];

    uint minidx = 0, maxidx = 0;

    for (uint r = 0; r < vec->rowSize; ++r)
    {
        if (vec->data[r * vec->colSize + 0] >= maxval)
        {
            maxval = vec->data[r * vec->colSize + 0];
            maxidx = r;
        }

        if (vec->data[r * vec->colSize + 0] <= minval)
        {
            minval = vec->data[r * vec->colSize + 0];
            minidx = r;
        }
    }

    printf("X[min = %d] = %.6lf | X[max = %d] = %.6lf\n",
           minidx, minval, maxidx, maxval);
}

void dampen(DMatrix *m)
{
    // multiply compatible matrix and vector

    float numpg = m->colSize, tmp;

    for (uint r = 0; r < m->rowSize; ++r)
    {
        for (uint c = 0; c < m->colSize; ++c)
        {
            tmp = m->data[r * m->colSize + c];
            m->data[r * m->colSize + c] = (1 - Q) * tmp + Q / numpg;
        }
    }
}

///////////////////////////////////////////////////////////////////////
__global__ void 
d_normalize(float *d_vec, int rowSize, int colSize, float *d_sum)
{
    // normalize vector using gpu
    int index = threadIdx.x;
    int stride = blockDim.x;

    if (threadIdx.x == 0) *d_sum = 0;
    __syncthreads();

    for (int r = index; r < rowSize; r += stride)
    {
        atomicAdd(d_sum, d_vec[r * colSize + 0]);  
    }

    __syncthreads();

    for (int r = index; r < rowSize; r += stride)
    {
        d_vec[r * colSize + 0] /= *d_sum;
    }
       
}

void vecNormalize(Vector *vec)
{
    // normalize the content of vector

    //kernel call
    float *d_vec, *d_sum;

    //allocate space on device
    cudaMalloc((void**)&d_sum, sizeof(float));
    cudaMalloc((void**)&d_vec, sizeof(float) * vec->colSize * vec->rowSize);

    cudaMemcpy(d_vec, vec->data,
        sizeof(float) * vec->colSize * vec->rowSize,
        cudaMemcpyHostToDevice);
    
    int blocksPerGrid = 1;
    int threadsPerBlock = MAX_BLOCK_SIZE;

    d_normalize<<<blocksPerGrid, threadsPerBlock>>>(d_vec, vec->rowSize, vec->colSize, d_sum);
    // printf("d_normalize error code: %s\n",cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(vec->data, d_vec, 
        sizeof(float) * vec->colSize * vec->rowSize,
        cudaMemcpyDeviceToHost);

    // deallocate space from device
    cudaFree(d_sum);
    cudaFree(d_vec);
}

__global__
void d_mult(float *vec, float *mat, float *out, const int colSize, const int rowSize)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
   
    float sum = 0.0;
    int vcolSize = 1;

    if(r < rowSize)
    {
        sum = 0.0;
        for(int c = 0; c < colSize; c++)
        {
            sum += mat[r * colSize + c] * vec[c * vcolSize + 0];
        }
        out[r * vcolSize + 0] = sum;
    }
 
}


Vector *matVec(DMatrix *m, Vector *vec)
{
    // multiply compatible matrix and vector
    // create and initialize at the pagerank Vector
    Vector *res = initVector(vec->rowSize);
    
    int threadsPerBlock = MAX_BLOCK_SIZE;
    int blocksPerGrid = m->rowSize / MAX_BLOCK_SIZE + 1;
    // int blocksPerGrid = 1;
    float *d_m, *d_vec, *d_res;

    //allocate space on device
    cudaMalloc((void**)&d_m, sizeof(float) * m->rowSize * m->colSize);
    cudaMalloc((void**)&d_vec, sizeof(float) * vec->rowSize * vec->colSize);
    cudaMalloc((void**)&d_res, sizeof(float) * res->rowSize * res->colSize);


    cudaMemcpy(d_m, m->data,
        sizeof(float)  * m->rowSize * m->colSize,
        cudaMemcpyHostToDevice);

    cudaMemcpy(d_vec, vec->data,
        sizeof(float) * vec->rowSize * vec->colSize,
        cudaMemcpyHostToDevice);

    cudaMemcpy(d_res, res->data,
        sizeof(float) * res->rowSize * res->colSize,
        cudaMemcpyHostToDevice);

    d_mult<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_m, d_res, m->colSize, m->rowSize);   
    // printf("d_mult error code: %s\n",cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(res->data, d_res, 
        sizeof(float) * res->rowSize * res->colSize,
        cudaMemcpyDeviceToHost);

    // deallocate space from device
  
    cudaFree(d_res);
    cudaFree(d_vec);
    cudaFree(d_m);

    // printDMatrix(res);


    vecNormalize(res);
    destroyDMatrix(vec);

    return res;
}
