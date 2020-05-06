/* Assignment 2 source code */

// #include <omp.h>
#include <stdio.h>
#include "libs/pagerankOMP.h"

int main(int argc, char *argv[])
{
    //reading number of pages from terminal
    uint numpg = (argc > 1) ? atoi(argv[1]) : 16;
    printf("number of pages = %d\n", numpg);

    #pragma omp parallel                   
    {
        printf("worker %d/%d ready to roll\n", omp_get_thread_num(), omp_get_num_threads());
    }  
    // create the Sparse matrix S
    DMatrix *tmp = initDMatrix(numpg);
    SMatrix *S = initSMatrix(numpg, tmp->data);
    destroyDMatrix(tmp);
    Vector* pgrkV2 = initVector(numpg);

    // display the S matrix
    // printf("S matrix = \n");
    // printSMatrix(S);

    //prints pagerank vector before matvec
    // printf("pagerank vector before web surfing\n");
    // printDMatrix(pgrkV2);

    /* timer */
    double startTime, endTime;

    startTime = omp_get_wtime();


    // apply matvec with dampening on for 1000 iterations
    for (uint iter = 0; iter < K; ++iter) {
        pgrkV2 = matVecSp(S, pgrkV2); // parallelized matVecDampn
        // printf("pagerank after iter %d \n", iter);
        // printDMatrix(pgrkV2);
    }

    if (numpg <= 16)
    { // print the page rank vector is small
        printf("pagerank vector after web surfing\n");
        printDMatrix(pgrkV2);
    }

    // display lowest and highest page ranks
    printf("page rank results: \n");
    minmaxPageRank(pgrkV2);

    endTime = omp_get_wtime();
    printf("\nruntime = %.16e\n", endTime - startTime);

    // garbage management
    destroySMatrix(S);
    destroyDMatrix(pgrkV2);

    return 0;
}