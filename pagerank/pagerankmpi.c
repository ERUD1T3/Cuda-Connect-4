/* Assignment 1 source code */

#include <stdio.h>
#include <stdlib.h>
#include "libs/dmatrix.h"
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
void vecNormalize(Vector *vec);                                       // normalize values of surfer values
Vector *matVec(DMatrix *mat, Vector *vec);                            // multiply compatible matrix and vector
void fillDMatrixMultProc(uint pid, uint npp, uint numpg, DMatrix *H); // transform H matrix into G (dampened) matrix

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Status status;
    uint pid, numprocs; // process id and number of processes

    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    double startTime, endTime;

    //reading number of pages from terminal
    uint numpg = (argc > 1) ? atoi(argv[1]) : 16;

    // printf("Worker %d/%d ready to roll, numpg= %d\n", pid + 1, numprocs, numpg);
    uint npp = numpg / numprocs; // determining the number of pages per processor

    // create the H matrix
    // DMatrix *H = initDMatrix(numpg);
    DMatrix *H = initDMatrixV(npp, numpg, 0.0);

    // create and initialize at the pagerank Vector
    Vector *pgrkV = initVectorP(npp, numpg);

    // display the H matrix
    // printf("pid = %d and here is my H matrix\n", pid);
    // printDMatrix(H);

    //prints pagerank vector before matvec
    // printf("pid = %d and pagerank vector before web surfing\n", pid);
    // printDMatrix(pgrkV);

    // fill partial H matrices based on mapping functions indices
    fillDMatrixMultProc(pid, npp, numpg, H);

    if (pid == 0)
    {
        startTime = MPI_Wtime();
    }

    dampen(H); //dampen own copy of matrix
    // apply matvec with dampening on for 1000 iterations
    for (uint iter = 0; iter < K; ++iter)
    {
        // allGather pgrkV to multiply to matvec local H
        double *tmp = malloc(sizeof(double) * npp);

        for (uint i = 0; i < npp; ++i)
        {
            tmp[i] = pgrkV->data[i][0];
        }

        // MPI_Gather(...)
        double *total = (double *)malloc(sizeof(double) * numpg);

        MPI_Allgather(tmp, npp, MPI_DOUBLE, total, npp, MPI_DOUBLE, MPI_COMM_WORLD);

        Vector *totalV = initVectorV(numpg, 0.0);
        for (uint i = 0; i < numpg; ++i)
        {
            totalV->data[i][0] = total[i];
        }

        pgrkV = matVec(H, totalV);
        //     printf("pagerank after iter %d\n", iter);
        //     printDMatrix(pgrkV);

        destroyDMatrix(totalV);
        free(tmp);
        free(total);
    }

    double *total = NULL;
    if (pid == 0)
        total = malloc(sizeof(double) * numpg);

    double *tmp = malloc(sizeof(double) * npp);

    for (uint i = 0; i < npp; ++i)
        tmp[i] = pgrkV->data[i][0];

    // MPI_Gather(...)
    MPI_Gather(tmp, npp, MPI_DOUBLE, total, npp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (pid == 0)
    {
        // printf("pagerank vector after web surfing\n");

        Vector *totalV = initVectorV(numpg, 0.0);
        for (uint i = 0; i < numpg; ++i)
        {
            totalV->data[i][0] = total[i];
        }

        if (numpg <= 16)
            printDMatrix(totalV);
        minmaxPageRank(totalV);

        endTime = MPI_Wtime();
        printf("\nruntime = %.16e\n", endTime - startTime);

        destroyDMatrix(totalV);
        free(total);
    }
    free(tmp);

    // garbage management
    destroyDMatrix(H);
    destroyDMatrix(pgrkV);

    MPI_Finalize();

    return 0;
}

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

Vector *matVec(DMatrix *mat, Vector *vec)
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

void fillDMatrixMultProc(uint pid, uint npp, uint numpg, DMatrix *H)
{
    // fill DMatrix in several processor

    // assume all partial Hs have been initialized to 0
    if (pid == locR(0, npp).pid)
    {
        H->data[locR(0, npp).locR][numpg - 1] = .5;
    }
    //    matrix->data[0][numpg - 1] = .5;

    if (pid == locR(1, npp).pid)
    {
        H->data[locR(1, npp).locR][0] = 1.0;
    }
    // H->data[1][0] = 1.0;

    for (uint r = 0, c = 1; r < numpg - 1; ++r, ++c)
    {
        if (pid == locR(r, npp).pid)
        {
            H->data[locR(r, npp).locR][c] = .5;
        }
        // H->data[r][c] = .5;
    }

    for (uint r = 2, c = 1; r < numpg; ++r, ++c)
    {
        if (pid == locR(r, npp).pid)
        {
            H->data[locR(r, npp).locR][c] = .5;
        }
        // H->data[r][c] = .5;
    }
}