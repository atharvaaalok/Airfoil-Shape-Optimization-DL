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
for i in range(len(airfoil_names)):
    X = X_list[i]
    L_by_D[i] = compute_L_by_D(X)
    print(i, np.isnan(L_by_D[i]))


finish_time = time.perf_counter()
print('Time Taken:', finish_time - start_time)

np.savetxt('100_repanel', L_by_D)