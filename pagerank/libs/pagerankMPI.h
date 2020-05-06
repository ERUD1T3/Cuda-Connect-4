// pagerank with MPI multithreading

#ifndef PAGERANKMPI_H
#define PAGERANKMPI_H


#include <stdio.h>
#include <stdlib.h>
#include "dmatrix.h"
#include "smatrix.h"
#include "mpi.h"

//constants
// #define Q .15  // dampening factor
#define K 1000 // number of matvec iterations
const double Q = .15;

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//function prototypes

/* non -parallel */
void minmaxPageRank(Vector *vec);
void dampen(DMatrix *mat);    

/* parallel */
void vecNormalize(Vector *vec);                      // normalize values of surfer values
Vector* matVec(DMatrix *mat, Vector *vec); // multiply compatible matrix and vector
Vector* matVecSp(SMatrix *mat, Vector *vec, uint npp);
void fillDMatrixMultProc(uint pid, uint npp, uint numpg, DMatrix* H);
                     // transform H matrix into G (dampened) matrix

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// definition of dense matrix object
void minmaxPageRank(Vector *vec)
{
    // return the max and min values in the vector, as well as their indices
     double minval = vec->data[0][0], maxval = vec->data[0][0];
    uint minidx = 0, maxidx = 0;

    for (uint r = 0; r < vec->numRow; ++r)
    {
        if (vec->data[r][0] >= maxval)
        {
            maxval = vec->data[r][0];
            maxidx = r;
        }

        if (vec->data[r][0] <= minval)
        {
            minval = vec->data[r][0];
            minidx = r;
        }
    }

    printf("X[min = %d] = %.6lf | X[max = %d] = %.6lf\n",
           minidx, minval, maxidx, maxval);
}

void dampen(DMatrix *mat)
{
    // multiply compatible matrix and vector

    uint numpg = mat->numCol;

    for (uint r = 0; r < mat->numRow; ++r)
        for (uint c = 0; c < mat->numCol; ++c)
            mat->data[r][c] = (1.0 - Q) * mat->data[r][c] + Q / numpg;

    // printf("Dampened : \n");
    // printDMatrix(mat);
    // return mat;
}



/////////////////////////////////////////////////////////////////////
void vecNormalize(Vector *vec)
{
    // normalize the content of vector to sum up to one
    // parallelized vecNormalize
    double loc_sum = 0.0;

    // forming local sum 
    for (uint r = 0; r < vec->numRow; ++r)
        loc_sum += vec->data[r][0];

    double glob_sum;
    // mpi allReduce sum of vec entries
    MPI_Allreduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    

    // divide your copy of x
    for (uint r = 0; r < vec->numRow; ++r)
        vec->data[r][0] /= glob_sum;
}

Vector* matVec(DMatrix *mat, Vector *vec)
{
    // multiply compatible matrix and vector

    Vector *res = initVectorV(mat->numRow, 0.0);
    // dampen(mat); //dampen own copy of matrix

    double tmp;
    
    for (uint r = 0; r < mat->numRow; ++r)
    {
        
        tmp = 0.0;
        // res->data[r][0] = 0.0;
  
        for (uint c = 0; c < mat->numCol; ++c)
        {
            tmp += mat->data[r][c] * vec->data[c][0];
        }

        res->data[r][0] = tmp;
    }

    vecNormalize(res);
    return res;
    
}

Vector* matVecSp(SMatrix *mat, Vector *vec, uint npp)
{
    Vector *res = initVectorV(npp, 0.0);

    double tmp;
    for (uint r = 0; r < mat->rowidxN - 1; ++r)
    {
        tmp = 0.0;
        // res->data[r][0] = 0.0;

        for (uint c = mat->rowidx[r]; c < mat->rowidx[r + 1]; ++c)
        {
            tmp += mat->nnzels[c] * vec->data[mat->colidx[c]][0];
        }

        // res->data[r][0] = tmp;
        res->data[r][0] = tmp * (1 - Q) + Q / vec->numRow;
    }

    vecNormalize(res);
    return res;
}


void fillDMatrixMultProc(uint pid, uint npp, uint numpg, DMatrix* H) {
    // fill DMatrix in several processor

    // assume all partial Hs have been initialized to 0
    if(pid == locR(0, npp).pid) {
        H->data[locR(0, npp).locR][numpg - 1] = .5;
    } 
    //    matrix->data[0][numpg - 1] = .5;

    if(pid == locR(1, npp).pid) {
        H->data[locR(1, npp).locR][0] = 1.0;
    }
    // H->data[1][0] = 1.0;

    for (uint r = 0, c = 1; r < numpg - 1; ++r, ++c){
        if(pid == locR(r, npp).pid) {
            H->data[locR(r, npp).locR][c] = .5;
        }
        // H->data[r][c] = .5;
    }

    for (uint r = 2, c = 1; r < numpg; ++r, ++c) {
        if(pid == locR(r, npp).pid) {
            H->data[locR(r, npp).locR][c] = .5;
        }
        // H->data[r][c] = .5;
    }
       
}

#endif // DENSE_MAT
