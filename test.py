from algos import factorize, text_to_inv, compose
from rewrite import *

tangle = text_to_inv("1:3',2:4,3:2',1':4'")

factors = factorize(tangle)
print(factors)
inv = compose(factors, 4)
print(inv)


init()
factors = rewrite(factors)

print(factors)