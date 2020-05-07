/* Assignment 1 source code */

// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "libs/dmatrix.h"

//constants
// #define Q .15  // dampening factor
#define K 1000 // number of matvec iterations
const double Q = .15;

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//function prototypes

/* non parallel */
void minmaxPageRank(Vector *vec);
void dampen(DMatrix *H); // transform H matrix into G (dampened) matrix

/* parallel */
void vecNormalize(Vector *vec);            // normalize values of surfer values
Vector *matVec(DMatrix *mat, Vector *vec); // multiply compatible matrix and vector

int main(int argc, char *argv[])
{
    //reading number of pages from terminal
    uint numpg = (argc > 1) ? atoi(argv[1]) : 16;

    printf("number of pages = %d\n", numpg);

    #pragma omp parallel
    {
        printf("worker %d/%d ready to roll\n", omp_get_thread_num() + 1, omp_get_num_threads());
    }

    /*timers*/
    double startTime, endTime;

    // create the H matrix
    DMatrix *H = initDMatrix(numpg);

    // create and initialize at the pagerank Vector
    Vector *pgrkV = initVector(numpg);

    // display the H matrix
    // printDMatrix(H);

    //prints pagerank vector before matvec
    // printf("pagerank vector before web surfing\n");
    // printDMatrix(pgrkV);

    startTime = omp_get_wtime();

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

    endTime = omp_get_wtime();
    printf("\nruntime = %.16e\n", endTime - startTime);

    // garbage management
    destroyDMatrix(H);
    destroyDMatrix(pgrkV);

    return 0;
}

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
            mat->data[r][c] = Q / numpg + (1.0 - Q) * mat->data[r][c];

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

    #pragma omp parallel for reduction(+: sum)
    for (uint r = 0; r < vec->numRow; ++r)
    {
        int myid = omp_get_thread_num();
        sum += vec->data[r][0];
        // printf("\nmyid= %d and sum= %.6lf\n", myid, sum);
    }

    #pragma omp parallel for
    for (uint r = 0; r < vec->numRow; ++r)
        vec->data[r][0] /= sum;
}

Vector *matVec(DMatrix *mat, Vector *vec)
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
