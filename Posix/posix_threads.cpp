#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <math.h>

#include "common.h"

static pthread_barrier_t barrier;

typedef struct {
	int M;
	int N;
	int i;
	int num_threads;
	double hx;
	double hy;
	double** w;
	double** w_old;
} pthrData;

void* threadFunc(void* thread_data) {
	pthrData* data = (pthrData*)thread_data;

	for (int f = data->i; f < data->N; f = f + data->num_threads) {
		for (int j = 1; j < data->M - 1; j++) {
			data->w[f][j] = ((data->w_old[f - 1][j] + data->w_old[f + 1][j]) * pow(data->hy, 2) + \
				(data->w_old[f][j - 1] + data->w_old[f][j + 1]) * pow(data->hx, 2) + \
				pow(data->hx, 2) * pow(data->hy, 2)) * 0.5 / (pow(data->hx, 2) + pow(data->hy, 2));
		}
	}
	
	return NULL;
}

double** posix_threads_poisson_executor(int N, int M, double D = 1.0, bool progress = true, int max_iter = 5000, double epsilon = 1e-6, int num_threads = 4) {
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

	pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
	pthrData* threadData = (pthrData*)malloc(num_threads * sizeof(pthrData));

	while (error >= epsilon)
	{
		error = 0.0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				w_old[i][j] = w[i][j];
			}
		}
		number_iteration++;

		for (int i = 0; i < num_threads; i++) {
			threadData[i].hx = hx;
			threadData[i].hy = hy;
			threadData[i].i = i+1;
			threadData[i].M = M;
			threadData[i].N = N-1;
			threadData[i].num_threads = num_threads;
			threadData[i].w = w;
			threadData[i].w_old = w_old;

			pthread_create(&(threads[i]), NULL, &threadFunc, &threadData[i]);
		}
		
		for (int d = 0; d < num_threads; d++) {
			pthread_join(threads[d], NULL);
		}

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
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

	free(threads);
	free(threadData);
	for (int i = 0; i < N; i++) {
		free(w_old[i]);
	}
	free(w_old);
	return w;
}