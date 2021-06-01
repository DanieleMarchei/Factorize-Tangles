from sys import argv
import rewrite as rew

_, term = argv
# term = "U 5,T 4,T 4,U 2,U 3,U 5,U 2,T 5,U 1,T 5,T 5,U 1,U 5,T 5,U 1,U 1,U 3,T 3,T 4,U 1,U 3,T 2,U 4,U 3,T 2,T 1,U 2,U 1,U 1,T 4"
rew.init()
res = rew.rewrite2(term, max_patience="auto")
print(res)