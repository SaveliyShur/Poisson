#pragma once
int test_solves(double** solves, double** test, int N, int M, int D, double max_error = 1e-5, bool print_solver = true);
void print_solution(double** solution, int N, int M);