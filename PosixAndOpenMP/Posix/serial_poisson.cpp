#include "stdio.h"
#include "omp.h"
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

#include "common.h"

double** serial_poisson_executor(int N, int M, double D, bool progress = true, int max_iter = 5000, double epsilon = 1e-9) {
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

	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < M - 1; j++) {
			w[i][j] = 0.5;
		}
	}

	double error = 1e6;
	double hy = 2.0 / (double)(N - 1);
	double hx = (2 * D) / (double)(M - 1);
	int number_iteration = 0;

	while (error >= epsilon)
	{
		error = 0.0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				w_old[i][j] = w[i][j];
			}
		}
		number_iteration++;

		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < M - 1; j++) {
				w[i][j] = ((w_old[i - 1][j] + w_old[i + 1][j]) * pow(hy, 2) + \
					(w_old[i][j - 1] + w_old[i][j + 1]) * pow(hx, 2) + \
					pow(hx, 2) * pow(hy, 2)) * 0.5 / (pow(hx, 2) + pow(hy, 2));
				error = fmax(error, fabs(w[i][j] - w_old[i][j]));
			}
		}

		if (number_iteration > max_iter) {
			break;
		}

		if (progress) {
			printf("N=%d, error=%.10lf\n", number_iteration, error);
		}
	}

	for (int i = 0; i < N; i++) {
		free(w_old[i]);
	}
	free(w_old);
	return w;
}