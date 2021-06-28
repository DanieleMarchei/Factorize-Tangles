import numpy as np
import maude

def init():
    ''' Initilizes the maude package.
    '''
    
    maude.init()
    maude.load('brauer.maude')


def rewrite(factor_list, max_patience = "auto"):
    ''' Rewrites the input term as an equivalent and (possibly) smaller factorization.
    Arguments:
        factor_list : the factor list to be reduced
        max_patience : integer, str (default: auto), how many steps inside a local minima are you willing to perform. If "auto" is equal to 1000 * length of term
    '''

    # covert factor list to Maude term
    term = []
    for factor in factor_list:
        term.append(f"{factor[0]} {factor[1:]}")

    term = ",".join(term)

    strategy_rule = "(delete ! | move)*"

    brauer = maude.getModule("REW-RULES")
    try:
        t = brauer.parseTerm(term)
    except:
        raise Exception("An error has occurred. Have you called init() before rewrite()?")
        

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
            # covert Maude term to factor list
            reduced_factor_list = []
            for factor in str(res[0]).split(","):
                reduced_factor_list.append(factor.replace(" ", ""))
            
            return rewrite(reduced_factor_list, max_patience)
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

    return factor_list