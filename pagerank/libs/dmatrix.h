#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
//typedefs
typedef unsigned int uint;
typedef struct dmatrix DMatrix;
typedef struct dmatrix Vector; // typedef for Vector based on DMatrix
typedef struct pair Pair;
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//function prototypes
DMatrix *initDMatrix(uint numpg); //initialize new dense matrix
DMatrix *initDMatrixV(uint rowsize, uint colsize, double val);
Vector *initVector(uint numpg);   // intialize a new vector
Vector *initVectorP(uint size, uint numpg);
Vector *initVectorV(uint size, double val);
void printDMatrix(DMatrix *dmat);
void fillDMatrix(DMatrix *mat, double val);
void fillDMatrixfromData(DMatrix *mat, double data[10][10]);
void destroyDMatrix(DMatrix *mat);
uint globR(uint pid, uint locr, uint npp);
Pair locR(uint globr, uint npp);

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// definition of dense matrix object

struct pair 
{
    uint pid, locR;
};

struct dmatrix
{
    //square matrix
    uint numRow, numCol;
    // probabilities contained in the matrix
    double **data;
};

DMatrix *initDMatrix(uint numpg)
{
    //initialize a new Dense matrix for rangking algorithm

    // create pointer to matrix
    DMatrix *matrix = (DMatrix *)malloc(sizeof(DMatrix));
    // matrix->numpg = numpg;
    matrix->numCol = numpg;
    matrix->numRow = numpg;

    // setting up data matrix to zeros
    matrix->data = (double **)malloc(numpg * sizeof(double *));
    for (uint r = 0; r < numpg; ++r)
        matrix->data[r] = (double *)malloc(numpg * sizeof(double));

    fillDMatrix(matrix, 0.0);
    // fillDMatrix(matrix, 1.0 / numpg);
    matrix->data[0][numpg - 1] = .5;
    matrix->data[1][0] = 1.0;

    for (uint r = 0, c = 1; r < numpg - 1; ++r, ++c)
        matrix->data[r][c] = .5;

    for (uint r = 2, c = 1; r < numpg; ++r, ++c)
        matrix->data[r][c] = .5;

    return matrix;
}


DMatrix *initDMatrixV(uint rowsize, uint colsize, double val)
{
    //initialize a new Dense matrix for rangking algorithm

    // create pointer to matrix
    DMatrix *matrix = (DMatrix *)malloc(sizeof(DMatrix));
    // matrix->numpg = numpg;
    matrix->numRow = rowsize;
    matrix->numCol = colsize;

    // setting up data matrix to zeros
    matrix->data = (double **)malloc(rowsize * sizeof(double *));
    for (uint r = 0; r < rowsize; ++r) {
        matrix->data[r] = (double *)malloc(colsize * sizeof(double));
    }

    fillDMatrix(matrix, val);

    return matrix;
}


void destroyDMatrix(DMatrix *mat)
{
    // detroy matrix object and free its memory
    for (uint r = 0; r < mat->numRow; ++r)
        free(mat->data[r]);
    free(mat->data);
    free(mat);
}

Vector *initVector(uint numpg)
{
    //initialize the surf vector
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    // vec->numpg = numpg;
    vec->numRow = numpg;
    vec->numCol = 1;

    // setting up data matrix to zeros
    vec->data = (double **)malloc(numpg * sizeof(double *));
    for (uint r = 0; r < numpg; ++r)
        vec->data[r] = (double *)malloc(sizeof(double));

    fillDMatrix(vec, 1.0 / numpg);
    return vec;
}

Vector *initVectorP(uint size, uint numpg)
{
    //initialize the surf vector
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    // vec->numpg = numpg;
    vec->numRow = size;
    vec->numCol = 1;

    // setting up data matrix to zeros
    vec->data = (double **)malloc(size * sizeof(double *));
    for (uint r = 0; r < size; ++r)
        vec->data[r] = (double *)malloc(sizeof(double));

    fillDMatrix(vec, 1.0 / numpg);
    return vec;
}


Vector *initVectorV(uint size, double val)
{
    //initialize the surf vector
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    // vec->numpg = numpg;
    vec->numRow = size;
    vec->numCol = 1;

    // setting up data matrix to zeros
    vec->data = (double **)malloc(size * sizeof(double *));
    for (uint r = 0; r < size; ++r)
        vec->data[r] = (double *)malloc(sizeof(double));

    fillDMatrix(vec, val);
    return vec;
}

void printDMatrix(DMatrix *dmat)
{
    // print the dense matrix
    printf("[\n");
    for (uint r = 0; r < dmat->numRow; ++r)
    {
        printf("[");
        for (uint c = 0; c < dmat->numCol; ++c)
        {
            printf(" %.6lf", dmat->data[r][c]);
        }
        printf(" ]\n");
    }
    printf("]\n");
}

void fillDMatrix(DMatrix *mat, double val)
{
    // fillDMatrix the content of a DMatrix to val specified
    for (uint r = 0; r < mat->numRow; ++r)
        for (uint c = 0; c < mat->numCol; ++c)
            mat->data[r][c] = val;
}

void fillDMatrixfromData(DMatrix *mat, double data[10][10])
{
    // fillDMatrix the content of a DMatrix to val specified
    for (uint r = 0; r < mat->numRow; ++r)
        for (uint c = 0; c < mat->numCol; ++c)
            mat->data[r][c] = data[r][c];
}

uint globR(uint pid, uint locr, uint npp) {
    // return global row index
    return pid*npp + locr;
}

Pair locR(uint globr, uint npp) {
    // return local row index and the pid 
    Pair res;
    res.pid = globr/npp;
    res.locR = globr%npp;

    return res;
}


#endif // DENSE_MATRIX_H