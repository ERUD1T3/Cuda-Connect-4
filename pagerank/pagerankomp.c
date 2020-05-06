/* Assignment 1 source code */

// #include <omp.h>
#include <stdio.h>
#include "libs/pagerankOMP.h"
#include "omp.h"

int main(int argc, char *argv[])
{
    //reading number of pages from terminal
    uint numpg = (argc > 1) ? atoi(argv[1]) : 16;

    printf("number of pages = %d\n", numpg);

    #pragma omp parallel                   
    {
        printf("worker %d/%d ready to roll\n", omp_get_thread_num(), omp_get_num_threads());
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
    for (uint iter = 0; iter < K; ++iter) {
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
}