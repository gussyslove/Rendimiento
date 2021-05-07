import pyximport
pyximport.install()
import functionE 
import funtionEcython
import numpy as np
from time import time

D = 5
N = 1500
X = np.array([np.random.rand(N) for d in range(D)]).T
beta = np.random.rand(N)
theta = 10

start_time=time()
functionE.rbf_network(X, beta, theta)
end_time=time()
time_python=end_time-start_time
print("Execution time (python)="+str(time_python))
start_time=time()
funtionEcython.rbf_network(X, beta, theta)
end_time=time()
time_cython=end_time-start_time
print("Execution time (cython)="+str(time_cython))
speedUp = round(time_python/time_cython, 3)
print("speedUp="+str(speedUp))