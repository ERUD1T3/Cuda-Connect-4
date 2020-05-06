#ifndef PAGERANK_H
#define PAGERANK_H

// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "dmatrix.h"
#include "smatrix.h"

//constants
#define K 1000 // number of matvec iterations

const double Q = .15;
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//function prototypes
void minmaxPageRank(Vector *vec);
void vecNormalize(Vector *vec);                      // normalize values of surfer values
void dampen(DMatrix *H);                         // transform H matrix into G (dampened) matrix
Vector* matVec(DMatrix *mat, Vector *vec); // multiply compatible matrix and vector
Vector* matVecSp(SMatrix *mat, Vector *vec);
// void matVecDampn(DMatrix *mat, Vector *vec, Vector *res); // multiply compatible matrix and vector

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

    double numpg = mat->numCol;

    double tmp;

    for (uint r = 0; r < mat->numRow; ++r) {
        for (uint c = 0; c < mat->numCol; ++c) {
            tmp = mat->data[r][c];
            mat->data[r][c] =  (1 - Q) * tmp + Q / numpg;
        }
    }
        

    // printf("Dampened : \n");
    // // printDMatrix(mat);
    // return mat;
}

///////////////////////////////////////////////////////////////////////
void vecNormalize(Vector *vec)
{
    // normalize the content of vector to sum up to one
    // parallelized vecNormalize
    double sum = 0.0;

    for (uint r = 0; r < vec->numRow; ++r)
        sum += vec->data[r][0];

    for (uint r = 0; r < vec->numRow; ++r)
        vec->data[r][0] /= sum;
}

Vector* matVec(DMatrix *mat, Vector *vec)
{
    // multiply compatible matrix and vector
    // create and initialize at the pagerank Vector
    Vector *res = initVector(vec->numRow);
    // dampen(mat);
    double tmp;
   
    // printf("Dampened : \n");
    // printDMatrix(mat);

    for (uint r = 0; r < mat->numRow; ++r)
    {
        tmp = 0.0;
        for (uint c = 0; c < mat->numCol; ++c)
        {
            tmp += mat->data[r][c] * vec->data[c][0];
        }

        res->data[r][0] = tmp;
    }

    vecNormalize(res);
    destroyDMatrix(vec);
    return res;
}

Vector* matVecSp(SMatrix *mat, Vector *vec)
{
    Vector *res = initVector(vec->numRow);
    double tmp;

    for (uint r = 0; r < mat->rowidxN - 1; ++r)
    {
        tmp = 0.0;
        for (uint c = mat->rowidx[r]; c < mat->rowidx[r + 1]; ++c)
        {
            tmp += mat->nnzels[c] * vec->data[mat->colidx[c]][0];
        }
        // dampening
        res->data[r][0] = tmp * (1-Q) + Q / vec->numRow;
    }

    vecNormalize(res);
    destroyDMatrix(vec);
    return res;
}





#endif // DENSE_MAT
