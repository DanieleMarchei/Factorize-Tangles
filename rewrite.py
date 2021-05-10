import numpy as np
import maude

def init():
    ''' Initilizes the maude package.
    '''
    
    maude.init()
    maude.load('brauer.maude')

def rewrite(term, max_patience = np.inf, strategy_rule = "(delete ! | move)*"):
    ''' Rewrites the input term as an equivalent and (possibly) smaller factorization.
    Arguments:
        term : the Maude term to be rewritten
        max_patience : integer (default: np.inf), how many steps inside a local minima are you willing to perform
    '''

    brauer = maude.getModule("REW-RULES")
    t = brauer.parseTerm(term)
    N = brauer.parseTerm(f"max_idx({term})")
    N.reduce()
    N = N.toInt() + 1
    factors_upper_bound = N * (N - 1) / 2

    strategy = brauer.parseStrategy(strategy_rule)
    

    min_factorization = (np.inf, None)

    results = t.srewrite(strategy, True)
    current_patience = max_patience
    out_of_patience = False
    for res in results:
        l = brauer.parseTerm(f"length({res[0]})")
        l.reduce()
        l = l.toInt()

        if l < min_factorization[0]:
            min_factorization = (l, res[0])
            current_patience = max_patience
        elif l > min_factorization[0]:
            break
        elif l <= factors_upper_bound:
            current_patience -= 1
        
        if current_patience <= 0 and max_patience > 0:
            out_of_patience = True
            break

    d = {
        "result" : min_factorization[1],
        "length" : min_factorization[0],
        "out of patience" : out_of_patience
    }

    return d
