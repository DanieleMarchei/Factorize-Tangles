import numpy as np
import maude
from sys import argv
from time import time

maude.init()
maude.load('brauer.maude')
brauer = maude.getModule("REW-RULES")
_, term = argv

t = brauer.parseTerm(term)
strategy = brauer.parseStrategy("(ruleRemove ! | ruleMove)*")

min_factorization = (np.inf, None)

start = time()

results = t.srewrite(strategy, True)
max_patience = 50000
current_patience = max_patience
for res in results:
    l = brauer.parseTerm(f"length({res[0]})")
    l.reduce()
    l = l.toInt()

    if l < min_factorization[0]:
        min_factorization = (l, res[0])
        current_patience = max_patience
    elif l > min_factorization[0]:
        break
    else:
        current_patience -= 1
    
    if current_patience <= 0 and max_patience > 0:
        print("Out of patience!")
        break


end = time()

elapsed_seconds = round(end - start, 2)

print(f"{min_factorization[1]}\nTime Elapsed : {elapsed_seconds} s")
