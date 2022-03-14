#include "stdio.h"
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

#include "common.h"

double** mpi_poisson_executor(int N, int M, double D = 1.0, bool progress = true, int max_iter = 5000, double epsilon = 1e-6, int num_threads = 4) {
	double** w = new double* [N];
	for (int i = 0; i < N; i++) {
		w[i] = new double[M];
	}

	double** w_old = new double* [N];
	for (int i = 0; i < N; i++) {
		w_old[i] = new double[M];
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			w[i][j] = 0.0;
			w_old[i][j] = 0.0;
		}
	}

	double error = 1e6;
	double hy = 2.0 / (double)(N - 1);
	double hx = (2 * D) / (double)(M - 1);
	int number_iteration = 0;

	for (int i = 0; i < N; i++) {
		free(w_old[i]);
	}
	free(w_old);
	return w;
}