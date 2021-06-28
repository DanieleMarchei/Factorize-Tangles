import numpy as np
import networkx as nx
import random


def inv_to_graph(inv):
    g = nx.Graph()
    g.add_edges_from(inv)
    
    return g


def prepare(g, N):
    mapping = {}
    for n in range(N):
        mapping[str(n+1)+"'"] = str(n+1) + "''"
        
    a = nx.relabel_nodes(g, mapping)
    mapping = {}
    for n in range(N):
        mapping[str(n+1)] = str(n+1) + "'"
    b = nx.relabel_nodes(a, mapping)
    return b

def to_tangle(g,N):
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

def get_inv(g,N):
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

def product(a,b,N):
    _b = prepare(b,N)
    c = nx.compose(a,_b)
    inv = to_tangle(c,N)
    nodes = [x for b in inv for x in b]
    edges = [(a[0],a[1]) for a in inv]
    res = nx.Graph()
    res.add_nodes_from(nodes)
    res.add_edges_from(edges)
    return res, inv

def generate_primes(N):
    names = []
    for n in range(N):
        names.append(str(n+1))
    for n in range(N):
        names.append(str(n+1) + "'")

    I =nx.Graph()
    I.add_nodes_from(names)
    for n in range(N):
        I.add_edge(str(n+1), str(n+1)+"'")


    primes_dict = {}

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
    
    return I, primes_dict

def compose(factors, N, start = "I"):
    I, primes_dict = generate_primes(N)

    if start == "I":
        composition = I
    else:
        composition = start
    for f in factors:
        composition, inv_composition = product(composition, primes_dict[f], N)
    
    return inv_composition

def is_hook(edge):
    return is_lower_hook(edge) or is_upper_hook(edge)

def is_lower_hook(edge):
    e1, e2 = edge
    return "'" in e1 and "'" in e2

def is_upper_hook(edge):
    e1, e2 = edge
    return "'" not in e1 and "'" not in e2

def is_transversal(edge):
    return not is_hook(edge)

def is_positive_transversal(edge):
    e1, e2 = edge_to_int(edge)
    return is_transversal(edge) and e1 > e2

def is_negative_transversal(edge):
    e1, e2 = edge_to_int(edge)
    return is_transversal(edge) and e1 < e2

def is_zero_transversal(edge):
    e1, e2 = edge_to_int(edge)
    return is_transversal(edge) and e1 == e2

def edge_to_int(edge):
    return [int(edge[0].replace("'","")), int(edge[1].replace("'",""))]

def size(edge):
    e1, e2 = edge_to_int(edge)
    return np.abs(e1-e2)

def is_full_cover(edge, h):
    if edge == h:
        return False
    e1, e2 = edge_to_int(edge)
    h1, h2 = edge_to_int(h)
    return (e1 < h1 and e2 > h2) or (e1 > h2 and e2 < h1)

def is_partial_cover(edge, h):
    if edge == h:
        return False
    e1, e2 = edge_to_int(edge)
    h1, h2 = edge_to_int(h)
    # if is_hook(edge):
    #     return (e1 == h1 and e2 >= h2) or (e1 <= h1 and e2 == h2)
    # else:
    #     return (e1 == h1 and e2 > h2) or (e1 == h2 and e2 < h1)
    return e1 == h1 or e1 == h2 or e2 == h1 or e2 == h2

def find_candidate(inv, h):
    #search for lower hooks that fully cover h
    for edge in inv:
        if is_lower_hook(edge) and is_full_cover(edge, h):
            return edge
    
    # if there is no lower hook, than take the fully covering edge with fewer intersecting edges
    few_penality_full_cover = (None, 999999)
    for edge in inv:
        if is_full_cover(edge, h):
            penality = edge_penality(inv, edge)
            if penality < few_penality_full_cover[1]:
                few_penality_full_cover = (edge, penality)
    
    if few_penality_full_cover[0] != None:
        return few_penality_full_cover[0]
    
    # if there is no biggest fully covering edge, take the partially covering one
    for edge in inv:
        if is_partial_cover(edge, h):
            return edge
    
    #this state should not be reachable
    raise Exception("Candidate not found")

def merge_lower_hook_with_edge(inv : list, h, edge):
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

    G = inv_to_graph(result)
    return get_inv(G, len(result))[0]

def merge_with_candidate(inv, h):
    edge = find_candidate(inv, h)
    return merge_lower_hook_with_edge(inv, h, edge)

def merge_edge_with_hook(edge, h):
    result = []
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
    return result

def is_cover(edge, h):
  return is_full_cover(edge,h) or is_partial_cover(edge, h)

def is_I(inv):
    for e in inv:
        e1,e2 = edge_to_int(e)
        if e1 != e2:
            return False
    return True

def is_T_tangle(inv):
    for e in inv:
        if not is_transversal(e):
            return False
    return True

def is_U_tangle(inv):
    for e in inv:
        if is_lower_hook(e) and size(e) == 1:
            return True
    return False

def is_H_tangle(inv):
    return not is_T_tangle(inv) and not is_U_tangle(inv)

def bottom_enumeration(inv):
    b = []
    for e in inv:
        _,e2 = edge_to_int(e)
        b.append(e2)
    
    return b

def find_swaps(b):
    factor_list = []
    for j in range(len(b)):
        for i in range(len(b) - 1 - j):
            if b[i] > b[i + 1]:
                b[i], b[i + 1] = b[i + 1], b[i]
                factor_list.append(f"T{i+1}")
    return factor_list

def factorizeT(inv):
    b = bottom_enumeration(inv)
    swaps = find_swaps(b)
    return swaps

def get_lower_hook_size_one(inv):
    for h in inv:
        if is_lower_hook(h) and size(h) == 1:
            return h

def get_smallest_lower_hook(inv):
    res = (None, np.inf)
    for h in inv:
        if is_lower_hook(h) and size(h) > 1:
            if size(h) < res[1]:
                res = (h, size(h))
    return res[0]

def get_covers(h, inv):
    covers = []
    for e in inv:
        if is_cover(e,h):
            covers.append(e)
    
    return covers

def factorizeU(inv):
    h = get_lower_hook_size_one(inv)
    i,_ = edge_to_int(h)
    C = get_covers(h, inv)

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
        return merge_lower_hook_with_edge(inv, single_cover, h), [f"U{i}"]
    
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
        return merge_lower_hook_with_edge(inv, single_cover, h), [f"U{i}"]
    
    # full cover lower hooks
    for cover in C:
        if is_lower_hook(cover):
            return merge_lower_hook_with_edge(inv, cover, h), [f"U{i}"]
    
    # pick one full cover at random
    full_covers = []
    for cover in C:
        if is_full_cover(cover,h):
            full_covers.append(cover)

    cover = random.choice(full_covers)
    return merge_lower_hook_with_edge(inv, cover, h), [f"U{i}"]

def interior_edges(inv, h):
    h1,h2 = edge_to_int(h)
    non_zero_int = []
    zero_int = []
    for edge in inv:
        if edge == h:
            continue

        if "'" not in edge[0] and "'" not in edge[1]:
            continue

        e1,e2 = edge_to_int(edge)
        if is_transversal(edge):
            if h1 < e2 < h2:
                if is_zero_transversal(edge):
                    zero_int.append(edge)
                else:
                    non_zero_int.append(edge)
        else:
            if e1 < h1 < e2 or  h1 < e2 < h2:
                non_zero_int.append(edge)
    
    return non_zero_int, zero_int
        

# def val(edge, h):
#     e1,e2 = edge_to_int(edge)
#     h1,h2 = edge_to_int(h)
#     if is_positive_transversal(edge) or is_lower_hook(edge) and e2 > h2:
#         return 1
    
#     if is_negative_transversal(edge) or is_lower_hook(edge) and e1 < h1:
#         return -1
    
#     if is_zero_transversal(edge):
#         return 0
    
#     raise Exception("Invalid edges")



def factorizeH(inv):
    h = get_smallest_lower_hook(inv)
    h1,h2 = edge_to_int(h)
    non_zero_int, zero_int = interior_edges(inv, h)

    interior = non_zero_int + zero_int
    val_lookup = {}
    for edge in interior:
        e1,e2 = edge_to_int(edge)
        if is_positive_transversal(edge):
            val_lookup[e2 - h1] = +1
        elif is_lower_hook(edge) and e2 > h2:
            val_lookup[e1 - h1] = +1
        elif is_negative_transversal(edge):
            val_lookup[e2 - h1] = -1
        elif is_lower_hook(edge) and e1 < h1:
            val_lookup[e2 - h1] = -1

    locations = [0]*size(h)
    for edge in non_zero_int:
        e1,e2 = edge_to_int(edge)
        if is_lower_hook(edge) and e2 > h2:
            locations[0] += val_lookup[e1 - h1]
        else:
            locations[0] += val_lookup[e2 - h1]

    locations[0] *= -1
    locations[0] += len(zero_int)

    best_location = (None, np.inf)
    for i in range(1,len(locations)):
        locations[i] = locations[i-1] + 2*val_lookup[i]
        if locations[i] < best_location[1]:
            best_location = (i+1, locations[i])
    
    j = best_location[0]
    L = []
    for i in range(h1,h1+j-1):
        L.append(f"T{i}")
    
    R = []
    for i in range(size(h)+h1-1,j+1,-1):
        R.append(f"T{i}")
    
    g = inv_to_graph(inv)
    LR = L + R
    result = compose(LR, len(inv), g)
    return result, L[::-1] + R[::-1]

def factorize(inv):
    if is_I(inv):
        return []
    
    if is_T_tangle(inv):
        return factorizeT(inv)
    
    if is_U_tangle(inv):
        new_inv, u = factorizeU(inv)
        return factorize(new_inv) + u
    
    if is_H_tangle(inv):
        new_inv, ts = factorizeH(inv)
        new_inv, u = factorizeU(new_inv)
        return factorize(new_inv) + u + ts