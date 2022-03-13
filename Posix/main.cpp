#include <iostream>
#include <cstdlib>
#include <omp.h>

#include "common.h"
#include "posix_threads.h"
#include "serial_poisson.h"
#include "open_mp_poisson.h"

int main()
{
	int N = 22;
	int M = 22;
	double D = 1.0;
	int thread = 2;

	omp_set_num_threads(thread);

	double t1 = omp_get_wtime();
	double** w = posix_threads_poisson_executor(N, M, D, false, 5000, 1e-6, 4);
	double t2 = omp_get_wtime();
	printf("Time = %.5lf s\n", t2 - t1);

	double** test = serial_poisson_executor(N, M, D, false);
	printf("Run tests:\n");
	int test_flag = test_solves(w, test, N, M, D, 1e-4, false);
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