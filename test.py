from algos import factorize, text_to_inv
from rewrite import *

tangle = text_to_inv("1:2',2:3',3:1',4:4',5:7',6:5',7:6'")

factors = factorize(tangle)

init()
factors = rewrite(factors)

print(factors)