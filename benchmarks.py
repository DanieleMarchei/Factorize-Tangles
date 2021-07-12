from numpy import tan
from algos import factorize,inv_to_text,compose, is_I
from rewrite import *
from random import choice
from time import time

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
    print(n)
    start = time()
    for _ in range(10*N):
        tangle = random_tangle(n)
        if is_I(tangle):
            continue
        
        factors = factorize(tangle)
        factors = rewrite(factors)
        minimal_tangle = compose(factors, n)
        if tangle != minimal_tangle:
            print(tangle)
            print(minimal_tangle)
            exit(1)
    end = time()
    print("time : ",end - start)