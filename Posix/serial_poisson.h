#pragma once
double** serial_poisson_executor(int N, int M, double D, bool progress = true, int max_iter = 5000, double epsilon = 1e-9);