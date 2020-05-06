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
void destroyDMatrix(DMatrix *mat);

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// definition of dense matrix object

struct pair 
{
    uint pid, locR;
};

struct dmatrix
{
    //square matrix
    uint rowSize, colSize;
    // probabilities contained in the matrix
    double *data;
    // row * width + col
};

DMatrix *initDMatrix(uint numpg)
{
    //initialize a new Dense matrix for rangking algorithm

    // create pointer to matrix
    DMatrix *m = (DMatrix *)malloc(sizeof(DMatrix));
    // matrix->numpg = numpg;
    m->colSize = numpg;
    m->rowSize = numpg;

    // setting up data matrix to zeros
    m->data = (double *)malloc(m->colSize * m->rowSize * sizeof(double));
    // for (uint r = 0; r < numpg; ++r)
    //     matrix->data[r] = (double *)malloc(numpg * sizeof(double));

    fillDMatrix(m, 0.0);
    // fillDMatrix(matrix, 1.0 / numpg);
    m->data[0*m->colSize + numpg - 1] = .5;
    m->data[1*m->colSize + 0] = 1.0;

    for (uint r = 0, c = 1; r < numpg - 1; ++r, ++c)
        m->data[r * m->colSize + c] = .5;

    for (uint r = 2, c = 1; r < numpg; ++r, ++c)
        m->data[r * m->colSize + c] = .5;

    return m;
}


DMatrix *initDMatrixV(uint rowsize, uint colsize, double val)
{
    //initialize a new Dense matrix for rangking algorithm

    // create pointer to matrix
    DMatrix *m = (DMatrix *)malloc(sizeof(DMatrix));
    // matrix->numpg = numpg;
    m->rowSize = rowsize;
    m->colSize = colsize;

    // setting up data matrix to zeros
    m->data = (double *) malloc(m->rowSize * m->colSize * sizeof(double));
    // for (uint r = 0; r < rowsize; ++r) {
    //     matrix->data[r] = (double *)malloc(colsize * sizeof(double));
    // }

    fillDMatrix(m, val);

    return m;
}


void destroyDMatrix(DMatrix *m)
{
    // detroy matrix object and free its memory
    // for (uint r = 0; r < mat->rowSize; ++r)
    //     free(mat->data[r]);
    free(m->data);
    free(m);
}

Vector *initVector(uint numpg)
{
    //initialize the surf vector
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    // vec->numpg = numpg;
    vec->rowSize = numpg;
    vec->colSize = 1;

    // setting up data matrix to zeros
    vec->data = (double *)malloc(vec->rowSize * vec->colSize * sizeof(double));
    // for (uint r = 0; r < numpg; ++r)
    //     vec->data[r] = (double *)malloc(sizeof(double));

    fillDMatrix(vec, 1.0 / numpg);
    return vec;
}

Vector *initVectorP(uint size, uint numpg)
{
    //initialize the surf vector
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    // vec->numpg = numpg;
    vec->rowSize = size;
    vec->colSize = 1;

    // setting up data matrix to zeros
    vec->data = (double *)malloc(vec->rowSize * vec->colSize * sizeof(double ));
    // for (uint r = 0; r < size; ++r)
    //     vec->data[r] = (double *)malloc(sizeof(double));

    fillDMatrix(vec, 1.0 / numpg);
    return vec;
}


Vector *initVectorV(uint size, double val)
{
    //initialize the surf vector
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    // vec->numpg = numpg;
    vec->rowSize = size;
    vec->colSize = 1;

    // setting up data matrix to zeros
    vec->data = (double *)malloc(vec->colSize * vec->rowSize * sizeof(double));
    // for (uint r = 0; r < size; ++r)
    //     vec->data[r] = (double *)malloc(sizeof(double));

    fillDMatrix(vec, val);
    return vec;
}

void printDMatrix(DMatrix *m)
{
    // print the dense matrix
    printf("[\n");
    for (uint r = 0; r < m->rowSize; ++r)
    {
        printf("[");
        for (uint c = 0; c < m->colSize; ++c)
        {
            printf(" %.6lf", m->data[r * m->colSize + c]);
        }
        printf(" ]\n");
    }
    printf("]\n");
}

void fillDMatrix(DMatrix *m, double val)
{
    // fillDMatrix the content of a DMatrix to val specified
    for (uint r = 0; r < m->rowSize; ++r)
        for (uint c = 0; c < m->colSize; ++c)
            m->data[r* m->colSize + c] = val;
}


#endif // DENSE_MATRIX_H