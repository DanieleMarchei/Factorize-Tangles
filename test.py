from factorizetangle import *

# mandatory initialization for the maude package
# run this before anything else
init()

inv = text_to_inv("1:3',2:4,3:2',1':4',5:7',6:6',7:5'")

factors = factorize_reduce(inv)
print(factors)

minimal_tangle = compose(factors, len(inv))

print(inv_to_text(minimal_tangle))