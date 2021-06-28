from algos import factorize,inv_to_text
from rewrite import *
from random import choice

def random_tangle(N):
    nodes = []
    for i in range(1, N+1):
        nodes.append(f"{i}")
    for i in range(1, N+1):
        nodes.append(f"{i}'")
    
    inv = []
    for i in range(N):
        d1 = nodes.pop(0)
        d2 = choice(nodes)
        nodes.remove(d2)
        inv.append([d1,d2])
    
    return inv


init()
N = 15
for n in range(3,N):
    for _ in range(10*N):
        tangle = random_tangle(n)
        print(inv_to_text(tangle))
        factors = factorize(tangle)
        factors = rewrite(factors)