import numpy as np
import maude

def init():
    ''' Initilizes the maude package.
    '''
    
    maude.init()
    maude.load('brauer.maude')

def rewrite(term, max_patience = np.inf):
    ''' Rewrites the input term as an equivalent and (possibly) smaller factorization.
    Arguments:
        term : the Maude term to be rewritten
        max_patience : integer, str (default: np.inf), how many steps inside a local minima are you willing to perform. If "auto" is equal to 1000 * length of term
    '''

    strategy_rule = "(delete ! | move)*"

    brauer = maude.getModule("REW-RULES")
    t = brauer.parseTerm(term)
    t_len = brauer.parseTerm(f"length({term})")
    t_len.reduce()
    t_len = t_len.toInt()

    if max_patience == "auto":
        max_patience = 1000 * t_len

    # calculate the max number of factors the term can have
    N = brauer.parseTerm(f"max_idx({term})")
    N.reduce()
    N = N.toInt() + 1
    factors_upper_bound = N * (N - 1) / 2

    strategy = brauer.parseStrategy(strategy_rule)
    
    min_factorization = (np.inf, None)

    # perform srew with depth first search
    results = t.srewrite(strategy, True)
    current_patience = max_patience
    out_of_patience = False
    for res in results:
        # calculate result length
        l = brauer.parseTerm(f"length({res[0]})")
        l.reduce()
        l = l.toInt()

        if l < min_factorization[0]:
            # save current best and restore current patience
            min_factorization = (l, res[0])
            current_patience = max_patience
            print(str(res[0]).replace(" ", ""))
        elif l > min_factorization[0]:
            # stop looping because we have reached the bottom of the tree
            break
        elif l <= factors_upper_bound:
            # do not start to lose patience until we have reached the upper bound
            current_patience -= 1
        
        if current_patience <= 0:
            # stop looping if patience was lost
            out_of_patience = True
            break

    d = {
        "result" : str(min_factorization[1]).replace(" ", ""),
        "length" : min_factorization[0],
        "out of patience" : out_of_patience
    }

    return d

def rewrite2(term, max_patience = np.inf):
    ''' Rewrites the input term as an equivalent and (possibly) smaller factorization.
    Arguments:
        term : the Maude term to be rewritten
        max_patience : integer, str (default: np.inf), how many steps inside a local minima are you willing to perform. If "auto" is equal to 1000 * length of term
    '''

    strategy_rule = "(delete ! | move)*"

    brauer = maude.getModule("REW-RULES")
    t = brauer.parseTerm(term)
    t_len = brauer.parseTerm(f"length({term})")
    t_len.reduce()
    t_len = t_len.toInt()

    if max_patience == "auto":
        max_patience = 1000 * t_len

    # calculate the max number of factors the term can have
    N = brauer.parseTerm(f"max_idx({term})")
    N.reduce()
    N = N.toInt() + 1
    factors_upper_bound = N * (N - 1) / 2

    strategy = brauer.parseStrategy(strategy_rule)
    
    # perform srew with depth first search
    results = t.srewrite(strategy, True)
    current_patience = max_patience
    for res in results:
        # calculate result length
        l = brauer.parseTerm(f"length({res[0]})")
        l.reduce()
        l = l.toInt()

        if l < t_len:
            return rewrite2(str(res[0]), max_patience)
        elif l > t_len:
            # stop looping because we have reached the bottom of the tree
            break
        elif l <= factors_upper_bound:
            # do not start to lose patience until we have reached the upper bound
            current_patience -= 1
        
        if current_patience <= 0:
            # stop looping if patience was lost
            break

    # no better solution was found
    return term