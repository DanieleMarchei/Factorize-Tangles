from sys import argv
import rewrite as rew

_, term = argv
rew.init()
res = rew.rewrite(term, max_patience="auto", strategy_rule="advanced")
print(res)