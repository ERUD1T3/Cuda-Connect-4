#ifndef PAGERANKOMP_H
#define PAGERANKOMP_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "dmatrix.h"
#include "smatrix.h"

//constants
// #define Q .15  // dampening factor
#define K 1000 // number of matvec iterations
const double Q = .15;

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//function prototypes

/* non parallel */
void minmaxPageRank(Vector *vec);
void dampen(DMatrix *H);                         // transform H matrix into G (dampened) matrix

/* parallel */
void vecNormalize(Vector *vec);                      // normalize values of surfer values
Vector* matVec(DMatrix *mat, Vector *vec); // multiply compatible matrix and vector
Vector* matVecSp(SMatrix *mat, Vector *vec);

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// definition of dense matrix object
void minmaxPageRank(Vector *vec)
{
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

    uint numpg = mat->numRow;

    for (uint r = 0; r < mat->numRow; ++r)
        for (uint c = 0; c < mat->numCol; ++c)
            mat->data[r][c] = Q / numpg + (1.0 - Q) * mat->data[r][c] ;

//  printf("Dampened : \n");
//     printDMatrix(mat);
    // return mat;
}


///////////////////////////////////////////////////////////////////////////////////////////

void vecNormalize(Vector *vec)
{
    // normalize the content of vector to sum up to one
    // parallelized vecNormalize
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (uint r = 0; r < vec->numRow; ++r) {
        int myid = omp_get_thread_num();
        sum += vec->data[r][0];
        // printf("\nmyid= %d and sum= %.6lf\n", myid, sum);
    }

       

    #pragma omp parallel for
    for (uint r = 0; r < vec->numRow; ++r)
        vec->data[r][0] /= sum;
}

Vector* matVec(DMatrix *mat, Vector *vec)
{
    // multiply compatible matrix and vector

    Vector *res = initVectorV(vec->numRow, 0.0);
    // fillDMatrix(res, 0.0);
    // dampen(mat);

    #pragma omp parallel for 
    for (uint r = 0; r < mat->numRow; ++r)
    {
        double tmp = 0.0;
        // res->data[r][0] = 0.0;
        // #pragma omp parallel for reduction(+:tmp)
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

    Vector *res = initVectorV(vec->numRow, 0.0);
    // fillDMatrix(res, 0.0);

    #pragma omp parallel for
    for (uint r = 0; r < mat->rowidxN - 1; ++r)
    {
        double tmp = 0.0;
        // res->data[r][0] = 0.0;

        #pragma omp parallel for reduction(+:tmp)
        for (uint c = mat->rowidx[r]; c < mat->rowidx[r + 1]; ++c)
        {
            tmp += mat->nnzels[c] * vec->data[mat->colidx[c]][0];
        }

        // res->data[r][0] = tmp;
        res->data[r][0] = tmp * (1-Q) + Q / vec->numRow;
    }

    vecNormalize(res);
    destroyDMatrix(vec);
    return res;
}



#endif // DENSE_MAT
