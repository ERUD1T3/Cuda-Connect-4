#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
// #include "sllist.h"

//typedefs
typedef unsigned int uint;
typedef struct smatrix SMatrix;

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//function prototypes
SMatrix *initSMatrix(uint numpg, double **data); // initialize a new sparse matrix
SMatrix *initSMatrixP(uint numpg, uint rowsize, uint colsize, double **data); 
void destroySMatrix(SMatrix *mat);
void printSMatrix(SMatrix *smat);
uint retNNZ(uint rowsize, uint colsize, double **data);

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// definition of dense matrix object
// definition of sparse matrix using CSR format
struct smatrix
{
    // size of vectors respectively
    uint nnzelsN, colidxN, rowidxN;
    // double *nnzels; // stores non zero elements of the matrix
    double *nnzels;
    uint *colidx, *rowidx; //partial sum of the num zero elements
};

SMatrix *initSMatrix(uint numpg, double **data)
{
    const uint nnz = 5 + 2 * (numpg - 3);
    SMatrix *newmat = (SMatrix *)malloc(sizeof(SMatrix));
    newmat->nnzelsN = nnz;
    newmat->colidxN = nnz;
    newmat->rowidxN = numpg + 1;
    newmat->nnzels = (double *)malloc(nnz * sizeof(double));
    newmat->rowidx = (uint *)malloc((numpg + 1) * sizeof(uint));
    newmat->colidx = (uint *)malloc(nnz * sizeof(uint));

    bool beganRow = false;
    uint nnz_counter = 0, colidx_counter = 0, rowidx_counter = 0;

    for (uint r = 0; r < numpg; ++r)
    {
        beganRow = true;
        for (uint c = 0; c < numpg; ++c)
        {
            if (data[r][c] != 0.0)
            {
                if (beganRow)
                { // begin a row
                    newmat->rowidx[rowidx_counter++] = nnz_counter;
                    beganRow = false;
                }
                // storing non zero elements
                newmat->nnzels[nnz_counter++] = data[r][c];
                // storing the columns of the non zero elements
                newmat->colidx[colidx_counter++] = c;
                // adding index of row
            }
        }
    }

    newmat->rowidx[numpg] = nnz;
    return newmat;
}

uint retNNZ(uint rowsize, uint colsize, double **data) {
    uint res = 0;
    for (uint r = 0; r < rowsize; ++r) 
    {
        for(uint c = 0; c < rowsize; ++c) 
        {
            if(data[r][c] != 0) {
                ++res;
            }  
        }
    }
    return res;
}


SMatrix *initSMatrixP(uint numpg, uint rowsize, uint colsize, double **data)
{
    const uint nnz = retNNZ(rowsize, colsize, data);
    SMatrix *newmat = (SMatrix *)malloc(sizeof(SMatrix));
    newmat->nnzelsN = nnz;
    newmat->colidxN = nnz;
    newmat->rowidxN = rowsize + 1;
    newmat->nnzels = (double *)malloc(nnz * sizeof(double));
    newmat->rowidx = (uint *)malloc((rowsize + 1) * sizeof(uint));
    newmat->colidx = (uint *)malloc(nnz * sizeof(uint));

    bool beganRow = false;
    uint nnz_counter = 0, colidx_counter = 0, rowidx_counter = 0;

    for (uint r = 0; r < rowsize; ++r)
    {
        beganRow = true;
        for (uint c = 0; c < colsize; ++c)
        {
            if (data[r][c] != 0.0)
            {
                if (beganRow)
                { // begin a row
                    newmat->rowidx[rowidx_counter++] = nnz_counter;
                    beganRow = false;
                }
                // storing non zero elements
                newmat->nnzels[nnz_counter++] = data[r][c];
                // storing the columns of the non zero elements
                newmat->colidx[colidx_counter++] = c;
                // adding index of row
            }
        }
    }

    newmat->rowidx[rowsize] = nnz;
    return newmat;
}

void destroySMatrix(SMatrix *mat)
{
    free(mat->nnzels);
    free(mat->colidx);
    free(mat->rowidx);
    free(mat);
}

void printSMatrix(SMatrix *smat)
{
    //prints the SMatrix
    printf("\non zero elements: ");
    for (uint i = 0; i < smat->nnzelsN; ++i)
        printf("%.6lf ", smat->nnzels[i]);
    printf("\ncolumn indices: ");
    for (uint i = 0; i < smat->colidxN; ++i)
        printf("%d ", smat->colidx[i]);
    printf("\nrow indices: ");
    for (uint i = 0; i < smat->rowidxN; ++i)
        printf("%d ", smat->rowidx[i]);

    printf("\n");
}
#endif // SPARSE_MATRIX_H