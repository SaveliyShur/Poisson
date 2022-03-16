# include <math.h>
# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>

double L = 1.0;			
int N = 10;		

double* u, * u_new;		

#define INDEX(i,j) ((N+2)*(i)+(j))

int my_rank;			

int* proc;		
int* i_min, * i_max;		
int* left_proc, * right_proc;	


int main(int argc, char* argv[]);
void allocate_arrays();
void jacobi(int num_procs, double f[]);
void make_domains(int num_procs);
double* make_source();
void timestamp();

int main(int argc, char* argv[])
{
    double change;
    double epsilon = 1.0E-09;
    double* f;
    char file_name[100];
    int i;
    int j;
    double my_change;
    int my_n;
    int n;
    int num_procs;
    int step;
    double* swap;
    double wall_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (1 < argc)
    {
        sscanf(argv[1], "%d", &N);
    }
    else
    {
        N = 10;
    }

    if (2 < argc)
    {
        sscanf(argv[2], "%lf", &epsilon);
    }
    else
    {
        epsilon = 1.0E-09;
    }
    if (3 < argc)
    {
        strcpy(file_name, argv[3]);
    }
    else
    {
        strcpy(file_name, "poisson_mpi.out");
    }

    allocate_arrays();
    f = make_source();
    make_domains(num_procs);

    step = 0;

    wall_time = MPI_Wtime();

    do
    {
        jacobi(num_procs, f);
        ++step;

        change = 0.0;
        n = 0;

        my_change = 0.0;
        my_n = 0;

        for (i = i_min[my_rank]; i <= i_max[my_rank]; i++)
        {
            for (j = 1; j <= N; j++)
            {
                if (u_new[INDEX(i, j)] != 0.0)
                {
                    my_change = my_change
                        + fabs(1.0 - u[INDEX(i, j)] / u_new[INDEX(i, j)]);

                    my_n = my_n + 1;
                }
            }
        }
        MPI_Allreduce(&my_change, &change, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Allreduce(&my_n, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (n != 0)
        {
            change = change / n;
        }
        if (my_rank == 0 && (step % 10) == 0)
        {
           // printf("  N = %d, n = %d, my_n = %d, Step %4d  Error = %g\n",
                //N, n, my_n, step, change);
        }
        swap = u;
        u = u_new;
        u_new = swap;
    } while (epsilon < change);

    wall_time = MPI_Wtime() - wall_time;
    if (my_rank == 0)
    {
        printf("\n");
        printf("  Wall clock time = %f secs\n", wall_time);
    }

    MPI_Finalize();

    free(f);

    if (my_rank == 0)
    {
        printf("\n");
        printf("POISSON_MPI:\n");
        printf("  Normal end of execution.\n");
        printf("\n");
        timestamp();
    }

    if (my_rank == 0) {
        printf("FIRST\n");
        for (int i = 0; i <= N+1; i++) {
            for (j = 0; j <= N+1; j++) {
                printf("%.3lf ", u[INDEX(i, j)]);
            }
            printf("\n");
        }
    }
    //printf("\n");
    //printf("\n");
    //printf("\n");
    //printf("\n");
    //printf("\n");


    if (my_rank == num_procs - 1) {
        for (int i = 0; i <= N+1; i++) {
            for (j = 0; j <= N+1; j++) {
                //printf("%.3lf ", u[INDEX(i, j)]);
            }
            //printf("\n");
        }
    }

    return 0;
}

void allocate_arrays()
{
    int i;
    int ndof;

    ndof = (N + 2) * (N + 2);

    u = (double*)malloc(ndof * sizeof(double));
    for (i = 0; i < ndof; i++)
    {
        u[i] = 0.0;
    }

    u_new = (double*)malloc(ndof * sizeof(double));
    for (i = 0; i < ndof; i++)
    {
        u_new[i] = 0.0;
    }

    return;
}

void jacobi(int num_procs, double f[])
{
    double h;
    int i;
    int j;
    MPI_Request request[4];
    int requests;
    MPI_Status status[4];

    h = L / (double)(N + 1);

    requests = 0;

    if (left_proc[my_rank] >= 0 && left_proc[my_rank] < num_procs)
    {
        MPI_Irecv(u + INDEX(i_min[my_rank] - 1, 1), N, MPI_DOUBLE,
            left_proc[my_rank], 0, MPI_COMM_WORLD,
            request + requests++);

        MPI_Isend(u + INDEX(i_min[my_rank], 1), N, MPI_DOUBLE,
            left_proc[my_rank], 1, MPI_COMM_WORLD,
            request + requests++);
    }

    if (right_proc[my_rank] >= 0 && right_proc[my_rank] < num_procs)
    {
        MPI_Irecv(u + INDEX(i_max[my_rank] + 1, 1), N, MPI_DOUBLE,
            right_proc[my_rank], 1, MPI_COMM_WORLD,
            request + requests++);

        MPI_Isend(u + INDEX(i_max[my_rank], 1), N, MPI_DOUBLE,
            right_proc[my_rank], 0, MPI_COMM_WORLD,
            request + requests++);
    }

    for (i = i_min[my_rank] + 1; i <= i_max[my_rank] - 1; i++)
    {
        for (j = 1; j <= N; j++)
        {
            u_new[INDEX(i, j)] =
                0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                    u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                    h * h * f[INDEX(i, j)]);
        }
    }

    MPI_Waitall(requests, request, status);

    i = i_min[my_rank];
    for (j = 1; j <= N; j++)
    {
        u_new[INDEX(i, j)] =
            0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                h * h * f[INDEX(i, j)]);
    }

    i = i_max[my_rank];
    if (i != i_min[my_rank])
    {
        for (j = 1; j <= N; j++)
        {
            u_new[INDEX(i, j)] =
                0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                    u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                    h * h * f[INDEX(i, j)]);
        }
    }

    return;
}

void make_domains(int num_procs)
{
    double d;
    double eps;
    int i;
    int p;
    double x_max;
    double x_min;

    proc = (int*)malloc((N + 2) * sizeof(int));
    i_min = (int*)malloc(num_procs * sizeof(int));
    i_max = (int*)malloc(num_procs * sizeof(int));
    left_proc = (int*)malloc(num_procs * sizeof(int));
    right_proc = (int*)malloc(num_procs * sizeof(int));

    eps = 0.0001;
    d = (N - 1.0 + 2.0 * eps) / (double)num_procs;

    for (p = 0; p < num_procs; p++)
    {

        x_min = -eps + 1.0 + (double)(p * d);
        x_max = x_min + d;

        for (i = 1; i <= N; i++)
        {
            if (x_min <= i && i < x_max)
            {
                proc[i] = p;
            }
        }
    }
    for (p = 0; p < num_procs; p++)
    {
        for (i = 1; i <= N; i++)
        {
            if (proc[i] == p)
            {
                break;
            }
        }
        i_min[p] = i;
        for (i = N; 1 <= i; i--)
        {
            if (proc[i] == p)
            {
                break;
            }
        }
        i_max[p] = i;

        left_proc[p] = -1;
        right_proc[p] = -1;

        if (proc[p] != -1)
        {
            if (1 < i_min[p] && i_min[p] <= N)
            {
                left_proc[p] = proc[i_min[p] - 1];
            }
            if (0 < i_max[p] && i_max[p] < N)
            {
                right_proc[p] = proc[i_max[p] + 1];
            }
        }
    }

    return;
}

double* make_source()
{
    double* f;
    int i;
    int j;
    int k;
    double q;

    f = (double*)malloc((N + 2) * (N + 2) * sizeof(double));

    for (i = 0; i < (N + 2) * (N + 2); i++)
    {
        f[i] = 1.0;
    }

    return f;
}

void timestamp()
{
# define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm* tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    printf("%s\n", time_buffer);

    return;
# undef TIME_SIZE
}