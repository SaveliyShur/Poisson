#include <iostream>
#include <cstdlib>
#include <omp.h>

#include "common.h"
#include "posix_threads.h"
#include "serial_poisson.h"
#include "open_mp_poisson.h"
#include "mpi_poisson.h"

int main()
{
	int N = 100;
	int M = 100;
	double D = 1.0;
	int thread = 8;

	omp_set_num_threads(thread);

	double t1 = omp_get_wtime();
	double** w = serial_poisson_executor(N, M, D, false, 100000, 1e-9);
	double t2 = omp_get_wtime();
	printf("Time = %.5lf s\n", t2 - t1);

	double** test = serial_poisson_executor(N, M, D, false);
	printf("Run tests:\n");
	int test_flag = test_solves(w, test, N, M, D, 1e-1, false);
	if (test_flag) {
		printf("Test success!\n");
	}
	else {
		printf("Test has failed!\n");
	}

	for (int i = 0; i < N; i++) {
		free(w[i]);
		free(test[i]);
	}
	free(w);
	free(test);

	std::string dummy;
    std::cout << "Enter to continue..." << std::endl;
    std::getline(std::cin, dummy);
    return 0;
}