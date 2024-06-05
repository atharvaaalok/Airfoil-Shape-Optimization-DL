import numpy as np

from airfoil_utils.compute_L_by_D import compute_L_by_D
import time


with open('airfoil_database/airfoil_names.txt', 'r') as f:
    airfoil_names = [names.strip() for names in f.readlines()]


X_list = []
for name in airfoil_names:
    X = np.loadtxt(f'airfoil_database/airfoils/{name}.dat')
    X_list.append(X)


start_time = time.perf_counter()


L_by_D = np.zeros(len(airfoil_names))
for i in range(400):
    print(i)
    X = X_list[i]
    L_by_D[i] = compute_L_by_D(X)


finish_time = time.perf_counter()
print('Time Taken:', finish_time - start_time)

np.savetxt('200_repanel', L_by_D)