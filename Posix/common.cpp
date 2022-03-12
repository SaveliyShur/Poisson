#include <math.h>
#include <stdio.h>

void print_solution(double** solution, int N, int M) {
	for (int d = 0; d < N; d++) {
		for (int v = 0; v < M; v++) {
			printf("%f ", solution[d][v]);
		}
		printf("\n");
	}
}

int test_solves(double** solves, double** test, int N, int M, int D, double max_error = 1e-5, bool print_solver = true) {
	int answer = 1;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			double r = test[i][j] - solves[i][j];
			if (fabs(test[i][j] - solves[i][j]) > max_error) {
				printf("Failed is %.10lf, i = %d, j = %d\n", fabs(test[i][j] - solves[i][j]), i, j);
				if (print_solver) {
					print_solution(test, N, M);
					printf("\n\n\n\n");
					print_solution(solves, N, M);
				}

				answer = 0;
			}
		}
	}

	return answer;
}