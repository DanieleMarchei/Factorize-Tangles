import numpy as np
import networkx as nx
       

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
    return is_transversal(edge) and e1 >= e2

def is_negative_transversal(edge):
    return not is_positive_transversal(edge)

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
    if is_hook(edge):
        return (e1 == h1 and e2 >= h2) or (e1 <= h1 and e2 == h2)
    else:
        return (e1 == h1 and e2 > h2) or (e1 == h2 and e2 < h1)

def are_intersecting_edges(edge1, edge2):
    e1_1, e1_2 = edge_to_int(edge1)
    e2_1, e2_2 = edge_to_int(edge2)
    if e1_1 > e2_1:
        return are_intersecting_edges(edge2, edge1)
    
    if is_transversal(edge1) and is_transversal(edge2):
        return e1_1 < e2_1 and e2_2 < e1_2
    if is_hook(edge1) and is_hook(edge2):
        return e1_1 < e2_1 < e1_2
    if is_upper_hook(edge1) and is_transversal(edge2):
        return e1_1 < e2_1 < e1_2
    if is_lower_hook(edge1) and is_transversal(edge2):
        return e1_1 < e2_2 < e1_2
    if is_transversal(edge1) and is_lower_hook(edge2):
        return e2_1 < e1_2 < e2_2
    if is_transversal(edge1) and is_upper_hook(edge2):
        return e2_1 < e1_1 < e2_2

    print(edge1, edge2)
    raise Exception("Error in edge type")

def n_intersecting_edges(inv, edge):
    n = 0
    for e in inv:
        if e == edge:
            continue
        if are_intersecting_edges(edge, e):
            n += 1
    return n

def edge_penality(inv, edge):
    penality = 0
    for e in inv:
        if e == edge:
            continue
        if is_hook(e):
            if are_intersecting_edges(edge, e):
                penality += 1
    return penality

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

def merge_lower_hook_with_all_edges(inv, h):
    for edge in inv:
        if edge == h:
            continue
        yield merge_lower_hook_with_edge(inv, h, edge)

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