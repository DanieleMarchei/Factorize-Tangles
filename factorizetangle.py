import numpy as np
import networkx as nx
import random
import maude


def _inv_to_graph(inv):
    g = nx.Graph()
    g.add_edges_from(inv)
    
    return g


def _prepare(g, N):
    mapping = {}
    for n in range(N):
        mapping[str(n+1)+"'"] = str(n+1) + "''"
        
    a = nx.relabel_nodes(g, mapping)
    mapping = {}
    for n in range(N):
        mapping[str(n+1)] = str(n+1) + "'"
    b = nx.relabel_nodes(a, mapping)
    return b

def _to_tangle(g,N):
    inv = []
    nodes = []
    for n in range(N):
        nodes.append(str(n+1))
    for n in range(N):
        nodes.append(str(n+1) + "''")

    done = []
    for n1 in nodes:
        for n2 in nodes:
            if n1 == n2:
                continue
            
            if n1 in done:
                continue
            
            if n2 in done:
                continue

            if nx.has_path(g, n1,n2):
                done.append(n1)
                done.append(n2)
                a = n1.replace("''", "'")
                b = n2.replace("''", "'")
                if [a,b] not in inv:
                    inv.append([a,b])

    return inv

def _get_inv(g,N):
    inv = []
    nodes = []
    for n in range(N):
        nodes.append(str(n+1))
    for n in range(N):
        nodes.append(str(n+1) + "'")

    done = []
    for n1 in nodes:
        for n2 in nodes:
            if n1 == n2:
                continue
            
            if n1 in done:
                continue
            
            if n2 in done:
                continue

            if nx.has_path(g, n1,n2):
                done.append(n1)
                done.append(n2)
                if [n1,n2] not in inv:
                    inv.append([n1,n2])

    return inv, str(inv)

def _product(a,b,N):
    _b = _prepare(b,N)
    c = nx.compose(a,_b)
    inv = _to_tangle(c,N)
    nodes = [x for b in inv for x in b]
    edges = [(a[0],a[1]) for a in inv]
    res = nx.Graph()
    res.add_nodes_from(nodes)
    res.add_edges_from(edges)
    return res, inv

def _generate_primes(N):
    names = []
    for n in range(N):
        names.append(str(n+1))
    for n in range(N):
        names.append(str(n+1) + "'")

    I =nx.Graph()
    I.add_nodes_from(names)
    for n in range(N):
        I.add_edge(str(n+1), str(n+1)+"'")


    primes_dict = {"I" : I}

    for u in range(1,N):
        U=nx.Graph()
        U.add_nodes_from(names)
        for i in range(1,u):
            U.add_edge(str(i), str(i)+"'")

        U.add_edge(str(u), str(u+1))
        U.add_edge(str(u)+"'", str(u+1)+"'")

        for i in range(u+2,N+1):
            U.add_edge(str(i), str(i)+"'")

        primes_dict["U"+str(u)] = U

    for t in range(1, N):
        T=nx.Graph()
        T.add_nodes_from(names)
        for i in range(1,t):
            T.add_edge(str(i), str(i)+"'")

        T.add_edge(str(t), str(t+1)+"'")
        T.add_edge(str(t+1), str(t)+"'")

        for i in range(t+2,N+1):
            T.add_edge(str(i), str(i)+"'")

        primes_dict["T"+str(t)] = T
    
    return primes_dict

def compose(factor_list, N):
    '''
    Composes the factors in the input factor list to obtain the tangle they create.
    '''
    primes_dict = _generate_primes(N)    
    composition = primes_dict["I"]
    for f in factor_list:
        composition, inv_composition = _product(composition, primes_dict[f], N)
    
    return inv_composition

def _compose(factor_list, N, start = "I"):
    primes_dict = _generate_primes(N)

    if start == "I":
        composition = primes_dict["I"]
    else:
        composition = start
    

    for f in factor_list:
        composition, inv_composition = _product(composition, primes_dict[f], N)
    
    return inv_composition

def is_hook(edge):
    '''
    Returns True if the input edge is a hook.
    '''
    return is_lower_hook(edge) or is_upper_hook(edge)

def is_lower_hook(edge):
    '''
    Returns True if the input edge is a lower hook.
    '''
    e1, e2 = edge
    return "'" in e1 and "'" in e2

def is_upper_hook(edge):
    '''
    Returns True if the input edge is a upper hook.
    '''
    e1, e2 = edge
    return "'" not in e1 and "'" not in e2

def is_transversal(edge):
    '''
    Returns True if the input edge is transversal.
    '''
    return not is_hook(edge)

def is_positive_transversal(edge):
    '''
    Returns True if the input edge is a positive transversal.
    '''
    e1, e2 = _edge_to_int(edge)
    return is_transversal(edge) and e1 > e2

def is_negative_transversal(edge):
    '''
    Returns True if the input edge is a negative transversal.
    '''
    e1, e2 = _edge_to_int(edge)
    return is_transversal(edge) and e1 < e2

def is_zero_transversal(edge):
    '''
    Returns True if the input edge is a zero transversal.
    '''
    e1, e2 = _edge_to_int(edge)
    return is_transversal(edge) and e1 == e2

def _edge_to_int(edge):
    return [int(edge[0].replace("'","")), int(edge[1].replace("'",""))]

def size(edge):
    '''
    Returns the size of the input edge
    '''
    e1, e2 = _edge_to_int(edge)
    return np.abs(e1-e2)

def is_full_cover(edge, h):
    '''
    Returns True if edge is a full cover for lower hook h.
    '''
    if edge == h:
        return False
    e1, e2 = _edge_to_int(edge)
    h1, h2 = _edge_to_int(h)
    return (e1 < h1 and e2 > h2) or (e1 > h2 and e2 < h1)

def is_partial_cover(edge, h):
    '''
    Returns True if edge is a partial cover for lower hook h.
    '''
    if edge == h:
        return False
    e1, e2 = _edge_to_int(edge)
    h1, h2 = _edge_to_int(h)
    if is_hook(edge):
        return (e1 == h1 and e2 >= h2) or (e1 <= h1 and e2 == h2)
    else:
        return (e1 == h1 and e2 > h2) or (e1 == h2 and e2 < h1)


def _merge_lower_hook_with_edge(inv : list, h, edge):
    result = inv[:]
    result.remove(edge)
    result.remove(h)
    if is_hook(edge):
        result.append([edge[0], h[0]])
        result.append([edge[1], h[1]])
    else:
        if is_positive_transversal(edge):
            result.append([edge[0], h[1]])
            result.append([edge[1], h[0]])
        else:
            result.append([edge[0], h[0]])
            result.append([h[1], edge[1]])

    G = _inv_to_graph(result)
    return _get_inv(G, len(result))[0]


def is_cover(edge, h):
    '''
    Returns True if edge is a cover for lower hook h.
    '''
    return is_full_cover(edge,h) or is_partial_cover(edge, h)

def is_I(inv):
    '''
    Returns True if the input tangle is the identity tangle and False otherwise.
    '''
    for e in inv:
        e1,e2 = _edge_to_int(e)
        if e1 != e2:
            return False
    return True

def is_T_tangle(inv):
    '''
    Returns True if the input tangle is a T-tangle and False otherwise.
    '''
    for e in inv:
        if not is_transversal(e):
            return False
    return True

def is_U_tangle(inv):
    '''
    Returns True if the input tangle is an U-tangle and False otherwise.
    '''
    for e in inv:
        if is_lower_hook(e) and size(e) == 1:
            return True
    return False

def is_H_tangle(inv):
    '''
    Returns True if the input tangle is a H-tangle and False otherwise.
    '''
    return not is_T_tangle(inv) and not is_U_tangle(inv)

def _bottom_enumeration(inv):
    b = []
    for e in inv:
        _,e2 = _edge_to_int(e)
        b.append(e2)
    
    return b

def _find_swaps(b,padding):
    factor_list = []
    for j in range(len(b)):
        for i in range(len(b) - 1 - j):
            if b[i] > b[i + 1]:
                b[i], b[i + 1] = b[i + 1], b[i]
                factor_list.append(f"T{i+1+padding}")
    return factor_list

def _factorizeT(inv,padding):
    b = _bottom_enumeration(inv)
    swaps = _find_swaps(b,padding)
    return swaps

def _get_lower_hook_size_one(inv):
    for h in inv:
        if is_lower_hook(h) and size(h) == 1:
            return h

def _get_smallest_lower_hook(inv):
    res = (None, np.inf)
    for h in inv:
        if is_lower_hook(h) and size(h) > 1:
            if size(h) < res[1]:
                res = (h, size(h))
    return res[0]

def _get_covers(h, inv):
    covers = []
    for e in inv:
        if is_cover(e,h):
            covers.append(e)
    
    return covers

def _factorizeU(inv,padding):
    h = _get_lower_hook_size_one(inv)
    i,_ = _edge_to_int(h)
    C = _get_covers(h, inv)

    # single partial cover
    single_cover = None
    for cover in C:
        if is_partial_cover(cover,h):
            if single_cover == None:
                single_cover = cover
            else:
                single_cover = None
                break
    
    if single_cover != None:
        return _merge_lower_hook_with_edge(inv, h, single_cover), [f"U{i+padding}"]
    
    # single full cover
    single_cover = None
    for cover in C:
        if is_full_cover(cover,h):
            if single_cover == None:
                single_cover = cover
            else:
                single_cover = None
                break
    
    if single_cover != None:
        return _merge_lower_hook_with_edge(inv, h, single_cover), [f"U{i+padding}"]
    
    # full cover lower hooks
    for cover in C:
        if is_lower_hook(cover):
            return _merge_lower_hook_with_edge(inv, h, cover), [f"U{i+padding}"]
    
    # pick one full cover at random
    full_covers = []
    for cover in C:
        if is_full_cover(cover,h):
            full_covers.append(cover)

    cover = random.choice(full_covers)
    return _merge_lower_hook_with_edge(inv, h, cover), [f"U{i+padding}"]

def _interior_edges(inv, h):
    h1,h2 = _edge_to_int(h)
    non_zero_int = []
    zero_int = []
    for edge in inv:
        if edge == h:
            continue

        if is_upper_hook(edge):
            continue

        e1,e2 = _edge_to_int(edge)
        if is_transversal(edge):
            if h1 < e2 < h2:
                if is_zero_transversal(edge):
                    zero_int.append(edge)
                else:
                    non_zero_int.append(edge)
        else:
            if e1 < h1 < e2 or h1 < e1 < h2:
                non_zero_int.append(edge)
    
    return non_zero_int, zero_int


def _factorizeH(inv,padding):
    h = _get_smallest_lower_hook(inv)
    h1,h2 = _edge_to_int(h)
    non_zero_int, zero_int = _interior_edges(inv, h)

    interior = non_zero_int + zero_int
    val_lookup = {}
    for edge in interior:
        e1,e2 = _edge_to_int(edge)
        if is_positive_transversal(edge):
            val_lookup[e2 - h1] = +1
        elif is_lower_hook(edge) and e2 > h2:
            val_lookup[e1 - h1] = +1
        elif is_negative_transversal(edge):
            val_lookup[e2 - h1] = -1
        elif is_lower_hook(edge) and e1 < h1:
            val_lookup[e2 - h1] = -1
        else:
            val_lookup[e2 - h1] = 0

    locations = [0]*size(h)
    for edge in non_zero_int:
        e1,e2 = _edge_to_int(edge)
        if is_lower_hook(edge) and e2 > h2:
            locations[0] += val_lookup[e1 - h1]
        else:
            locations[0] += val_lookup[e2 - h1]

    locations[0] *= -1
    locations[0] += len(zero_int)

    best_location = (0, locations[0])
    for i in range(1,len(locations)):
        locations[i] = locations[i-1] + 2*val_lookup[i]
        if locations[i] < best_location[1]:
            best_location = (i+1, locations[i])
    
    j = best_location[0]
    L = []
    for i in range(h1,h1+j-1):
        L.append(f"T{i}")
    
    R = []
    for i in range(size(h)+h1-1,j+h1-1,-1):
        R.append(f"T{i}")
    
    g = _inv_to_graph(inv)
    LR = L + R
    result = _compose(LR, len(inv), g)
    L = []
    for i in range(h1,h1+j-1):
        L.append(f"T{i+padding}")
    
    R = []
    for i in range(size(h)+h1-1,j+h1-1,-1):
        R.append(f"T{i+padding}")
    return result, L[::-1] + R[::-1]

def factorize(inv):
    '''
    Returns a factor list for every partition of the input tangle.
    '''
    return _factorize(inv, 0)

def _factorize(inv, original_padding):
    partitions = get_partitions(inv)
    factor_lists = []
    for tangle, padding in partitions:
        factors, is_optimal = _factorize_impl(tangle, padding + original_padding)
        if len(factors) != 0:
            factor_lists.append((factors, is_optimal))
    
    return factor_lists

def _factorize_impl(inv, padding, is_optimal = True):

    if is_I(inv):
        return [], is_optimal
    
    if is_T_tangle(inv):
        return _factorizeT(inv, padding), is_optimal
    
    if is_U_tangle(inv):
        new_inv, u = _factorizeU(inv, padding)
        res = _factorize_impl(new_inv, padding, False)
        res[0].append(u)
        return res
    
    # is H tangle by exclusion
    new_inv, ts = _factorizeH(inv, padding)
    new_inv, u = _factorizeU(new_inv, padding)
    res = _factorize_impl(new_inv, padding, False)
    res[0].extend(u + ts)
    return res

def text_to_inv(text):
    '''
    Returns a list representation of a tangle written in a string representation 
    '''
    inv = []
    pairs = text.split(",")
    for pair in pairs:
        a,b = pair.split(":")
        inv.append([a,b])
    
    return inv

def inv_to_text(inv):
    '''
    Returns a string representation of a tangle written in a list representation 
    '''
    s = ""
    for i, (a,b) in enumerate(inv):
            end = ","
            if i == len(inv) - 1:
                end = ""
            s += f"{a}:{b}{end}"
    return s

def _is_gap_crossed(inv, n):
    for edge in inv:
        e1, e2 = _edge_to_int(edge)
        if e1 <= n and e2 > n or e1 >= n and e2 < n:
            return True
    return False

def get_partitions(inv):
    '''
    Returns the partitions of the input tangle.
    '''
    parts = [[1, None]]
    n = len(inv)
    for i in range(1, n):
        if not _is_gap_crossed(inv, i):
            parts[-1][-1] = i
            parts.append([i+1, None])
    
    parts[-1][-1] = n
    
    partitions_list = []
    for p1,p2 in parts:
        partition = []
        for edge in inv:
            e1,e2 = _edge_to_int(edge)
            if p1 <= e1 and e2 <= p2:
                e1 = f"{(e1 - p1 + 1)}" if "'" not in edge[0] else f"{(e1 - p1 + 1)}'"
                e2 = f"{(e2 - p1 + 1)}" if "'" not in edge[1] else f"{(e2 - p1 + 1)}'"
                partition.append((e1,e2))

        partitions_list.append((partition, p1-1))
    
    return partitions_list


def init():
    ''' Initilizes the maude package.
    '''
    
    maude.init()
    maude.load('brauer.maude')


def reduce(factor_list, max_patience = "auto"):
    ''' Reduces the input term as an equivalent and (possibly) smaller factorization. \n
    Arguments:
        factor_list : the factor list to be reduced \n
        max_patience : integer, str (default: auto), how many steps inside a local minima are you willing to perform. If "auto" is equal to 1000 * length of term
    '''

    factor_list = [factor for factor in factor_list if factor != "I"]

    # covert factor list to Maude term
    term = []
    for factor in factor_list:
        if factor == "I": continue

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
            
            return reduce(reduced_factor_list, max_patience)
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

def factorize_reduce(inv, max_patience = "auto"):
    '''
    Returns the minimal factor list for the input tangle. \n
    Arguments:
        inv : the tangle to be factorized in list representation \n
        max_patience : integer, str (default: auto), how many steps inside a local minima are you willing to perform. If "auto" is equal to 1000 * length of term
    '''    
    factor_lists = factorize(inv)
    final_factor_list = []
    for factor_list, is_optimal in factor_lists:
        reduced_factor_list = factor_list
        if not is_optimal:
            reduced_factor_list = reduce(factor_list, max_patience)
        final_factor_list.extend(reduced_factor_list)
    
    return final_factor_list