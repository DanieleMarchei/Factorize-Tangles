from algos import compose
import rewrite as rew
from random import randint

rew.init()
n_factors = 4
max_upper_bound = 5

for i in range(1000):
    factors = []
    for n in range(n_factors):
        r = randint(1, max_upper_bound)
        if randint(1,2) == 1:
            factors.append(f"U {r}")
        else:
            factors.append(f"T {r}")

    s = ",".join(factors)

    res_remove_move = rew.rewrite(s)
    res_remove_only = rew.rewrite(s, strategy_rule="delete !")
    if res_remove_move["length"] != res_remove_only["length"]:
        print(s)
        print(res_remove_move["result"])
        print(res_remove_only["result"])
        print("---------")
        break
