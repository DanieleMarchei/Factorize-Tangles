from algos import *
from rewrite import *

tangle = text_to_inv("1:3',2:4,3:2',1':4'")
factors = factorize(tangle)
print(factors)

init()
factors = rewrite(factors)

print(factors)