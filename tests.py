from algos import *
from rewrite import rewrite2, init

def text_to_inv(text):
    #inv = 1:2,3:4,5:6
    inv = []
    pairs = text.split(",")
    for pair in pairs:
        a,b = pair.split(":")
        inv.append([a,b])
    
    return inv


# x = text_to_inv("1:4',2:3,4:1',5:3',6:5',2':6'")
x = text_to_inv("1:4',2:3',3:2',4:5,1':5'")
factors = factorize(x)
print(factors)
init()
term = rewrite2(factors)
print(term)