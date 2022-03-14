#pragma once
double** mpi_poisson_executor(int N, int M, double D = 1.0, bool progress = true, int max_iter = 5000, double epsilon = 1e-6, int num_threads = 4);