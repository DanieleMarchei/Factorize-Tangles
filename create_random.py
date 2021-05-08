from random import randint
from sys import argv

_, n_factors, max_upper_bound = argv
n_factors = int(n_factors)
max_upper_bound = int(max_upper_bound)

factors = []
for n in range(n_factors):
    r = randint(1, max_upper_bound)
    if randint(1,2) == 1:
        factors.append(f"U {r}")
    else:
        factors.append(f"T {r}")

s = ",".join(factors)
print(s)
