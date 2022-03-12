#include "stdio.h"
#include "omp.h"
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

#include "common.h"

double** open_mp_poisson_executor(int N, int M, double D = 1.0, bool progress = true, int max_iter = 20000, double epsilon = 1e-9) {
	double** w = new double* [N];
	for (int i = 0; i < N; i++) {
		w[i] = new double[M];
	}

	double** w_old = new double* [N];
	for (int i = 0; i < N; i++) {
		w_old[i] = new double[M];
	}

	double** err = new double* [N];
	for (int i = 0; i < N; i++) {
		err[i] = new double[M];
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			w[i][j] = 0.0;
			w_old[i][j] = 0.0;
			err[i][j] = 0.0;
		}
	}

	double error = 1e6;
	double hy = 2.0 / (double)(N - 1);
	double hx = (2 * D) / (double)(M - 1);
	int number_iteration = 0;
	{
		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < M - 1; j++) {
				w[i][j] = 0.5;
			}
		}

		while (error >= epsilon)
		{
			int i, j;
			error = 0;
			#pragma omp parallel shared(error, w, w_old, hy, hx) private(i, j) 
			{
				for (int di = 0; di < 50; di++) {
					#pragma omp for
					for (i = 0; i < N; i++) {
						for (j = 0; j < M; j++) {
							w_old[i][j] = w[i][j];
						}
					}
					#pragma omp barrier
					#pragma omp for
					for (i = 1; i < N - 1; i++) {
						for (j = 1; j < M - 1; j++) {
							w[i][j] = ((w_old[i - 1][j] + w_old[i + 1][j]) * pow(hy, 2) + \
								(w_old[i][j - 1] + w_old[i][j + 1]) * pow(hx, 2) + \
								pow(hx, 2) * pow(hy, 2)) * 0.5 / (pow(hx, 2) + pow(hy, 2));
							err[i][j] = fabs(w[i][j] - w_old[i][j]);
						}
					}
					#pragma omp barrier
				}
			}

			for (i = 0; i < N; i++) {
				for (j = 0; j < M; j++) {
					error = fmax(error, err[i][j]);
				}
			}

			number_iteration = number_iteration + 50;
			if (number_iteration > max_iter) {
				break;
			}

			if (progress) {
				printf("N=%d, error=%.10lf\n", number_iteration, error);
			}
		}
	}


	for (int i = 0; i < N; i++) {
		free(w_old[i]);
	}
	free(w_old);

	for (int i = 0; i < N; i++) {
		free(err[i]);
	}
	free(err);
	return w;
}