#pragma once
double** open_mp_poisson_executor(int N, int M, double D = 1.0, bool progress = true, int max_iter = 5000, double epsilon = 1e-9);