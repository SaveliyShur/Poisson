from mpi4py import MPI
import math
import numpy as np

L = 1.0
N = 100
epsilon = 1e-5

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_procs = comm.Get_size()

u = np.zeros((N + 2) * (N + 2),  dtype='float64')
u_new = np.zeros((N + 2) * (N + 2),  dtype='float64')
f = np.ones((N + 2) * (N + 2),  dtype='float64')
proc = np.zeros(N + 2,  dtype='int64')
i_min = np.zeros(num_procs,  dtype='int64')
i_max = np.zeros(num_procs,  dtype='int64')
left_proc = np.zeros(num_procs,  dtype='int64')
right_proc = np.zeros(num_procs,  dtype='int64')

def INDEX(i, j): return ((N+2)*(i)+(j))

def make_domains(num_procs):
    eps = 0.0001
    d = (N - 1.0 + 2.0 * eps) / float(num_procs)

    for p in range(num_procs):
        x_min = -eps + 1.0 + float(p * d)
        x_max = x_min + d
        for i in range(1, N + 1):
            if x_min <= i and i < x_max:
                proc[i] = p

    for p in range(num_procs):
        for i in range(1, N + 1):
            if proc[i] == p:
                i_min[p] = i
                break
        for i in range(N, 0, -1):
            if proc[i] == p:
                i_max[p] = i
                break
        left_proc[p] = -1;
        right_proc[p] = -1;

        if proc[p] != -1:
            if 1 < i_min[p] and i_min[p] <= N:
                left_proc[p] = proc[i_min[p] - 1]
            if 0 < i_max[p] and i_max[p] < N:
                right_proc[p] = proc[i_max[p] + 1]

    return

def jacobi(num_procs):
    h = L / float(N + 1)
    request = []

    if left_proc[my_rank] >= 0 and left_proc[my_rank] < num_procs:
        req = comm.Irecv(u[INDEX(i_min[my_rank] - 1, 1): INDEX(i_min[my_rank] - 1, 1) + N], left_proc[my_rank])
        request.append(req)
        # print("Request 1 from {0}".format(my_rank))

        req2 = comm.Isend(u[INDEX(i_min[my_rank], 1): INDEX(i_min[my_rank], 1) + N], left_proc[my_rank])
        request.append(req2)
        # print("Request 2 from {0}".format(my_rank))
            
    if right_proc[my_rank] >= 0 and right_proc[my_rank] < num_procs:
        req3 = comm.Irecv(u[INDEX(i_max[my_rank] + 1, 1): INDEX(i_max[my_rank] + 1, 1) + N], right_proc[my_rank])
        request.append(req3)
        # print("Request 3 from {0}".format(my_rank))

        req4 = comm.Isend(u[INDEX(i_max[my_rank], 1): INDEX(i_max[my_rank], 1) + N], right_proc[my_rank])
        request.append(req4)
        # print("Request 4 from {0}".format(my_rank))
    
    # print("Start solution from {0}".format(my_rank))
    for i in range(i_min[my_rank] + 1, i_max[my_rank]):
        for j in range(1, N + 1):
            u_new[INDEX(i, j)] = 0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                    u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                    h * h * f[INDEX(i, j)])
    
    # print("Start wait from {0}".format(my_rank))
    MPI.Request.waitall(request)
    # print("End wait from {0}".format(my_rank))

    i = i_min[my_rank]
    for j in range(1, N + 1):
        u_new[INDEX(i, j)] = 0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                h * h * f[INDEX(i, j)])

    i = i_max[my_rank]
    if i != i_min[my_rank]:
        for j in range(1, N + 1):
            u_new[INDEX(i, j)] = 0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                    u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                    h * h * f[INDEX(i, j)])
    return

make_domains(num_procs)

step = 0

wall_time = MPI.Wtime()

while True:
    jacobi(num_procs)
    step = step + 1

    change = np.zeros(1,  dtype='float64')
    n = np.zeros(1,  dtype='int64')
    my_change = np.zeros(1,  dtype='float64')
    my_n = np.zeros(1,  dtype='int64')

    for i in range(i_min[my_rank], i_max[my_rank] + 1):
        for j in range(1, N + 1):
            if (u_new[INDEX(i, j)] != 0.0):
                my_change[0] = my_change[0] + abs(1.0 - u[INDEX(i, j)] / u_new[INDEX(i, j)])
                my_n[0] = my_n[0] + 1

    comm.Allreduce(my_change, change, op=MPI.SUM)
    comm.Allreduce(my_n, n, op=MPI.SUM)

    if n[0] != 0:
        change[0] = change[0] / n[0]
    if my_rank == 0 and step % 200 == 0:
        print("N={0}, n = {1}, my_n = {2}, Step {3} Error {4}".format(N, n[0], my_n[0], step, change[0]))

    swap = u
    u = u_new
    u_new = swap

    if epsilon > change[0]:
        break

wall_time = MPI.Wtime() - wall_time

if my_rank == 0:
    print("Wall clock time = {0} secs".format(wall_time))

# print("I min = {0}", i_min[my_rank])
# print("I max = {0}", i_max[my_rank])