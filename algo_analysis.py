"""
Problem 5.
Let the function f(n) be defined as f(0) = 0,f(1) = 1,f(n) = (f(n−1)+f(n− 2))(mod 331) for all n > 1,
where a(mod b) is the remainder when a is divided by b (for example, 15(mod 6) = 3 since 15 = 2×6+3).
Write a C++ or Python program to implement the function f(.):
i. It follows the recursion f(n) = (f(n − 1) + f(n −2))(mod 331) to compute f(n).
ii. It computes f(n) without recursive call.
iii. Test each implementation at n = 20,30,40,50,100,10000, and 100000.
iv. Explain the difference between test results of the two implementations from the compiler and algorithm complexity point of view.
We give an example to compute function s(n) = 1+2+···+n with two different implementations in C++.
The first uses recursion that function s(.) calls itself, but the second does not use recursion. int s(int n)
"""
import time
import matplotlib.pyplot as plt
from numba import jit

N = [20, 30, 40, 50, 100, 10000, 100000]
recursion_runtime = []
iteration_runtime = []

@jit(nopython=True)
def recursion(n):
    if n==0:
        return 0
    if n==1:
        return 1
    return (recursion(n-1) + recursion(n-2))%331

@jit(nopython=True)
def fib_iter(n):
    a = 0
    b = 1
    c = 0
    for i in range(2, n+1):
        c = (a + b) % 331
        a = b
        b = c
    return b.item()

for n in N:
    start = time.perf_counter()
    recursion(n)
    end = time.perf_counter()
    recursion_runtime.append(end - start)
    start = time.perf_counter()
    fib_iter(n)
    end = time.perf_counter()
    iteration_runtime.append(end - start)
    print(f'Runtime for {n} completed!')

plt.title('Comparison between recusrion and iteration')
plt.xlabel('Value of n')
plt.ylabel('Time')
plt.plot(N, recursion_runtime, label='Recursion Runtime')
plt.plot(N, iteration_runtime, label='Iteration Runtime')
plt.xticks(N, labels=['20', '30', '40', '50', '100', '1000', '10000'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()