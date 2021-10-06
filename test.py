from factorizetangles import *

# mandatory initialization for the maude package
# run this before anything else
init()

# converts a text representation of the tangle into a list representation
inv = text_to_inv("1:3',2:4,3:2',5:7',6:6',7:5',1':4'")

# factorizes the tangle and tries to find the minimal factorization
factors = factorize_reduce(inv)
print(f"factors = {factors}")

latex = to_latex(factors)
print("latex = " + latex)

# find all equivalent minimal factor lists
equivalent_factorizations = find_equivalent(factors)
print("Or equivalently")
counter = 0
for equivalent in equivalent_factorizations:
    if counter < 10:
        print(",".join(equivalent))
        counter += 1 
    else:
        break


# compose the minimal factor list in order to obtain the original tangle
minimal_tangle = compose(factors, len(inv))

print("Composed factors")
print(inv_to_text(minimal_tangle))
