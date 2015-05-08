from __future__ import division
import csv
import time
import math
import random
import networkx as nx

#=======================================================================
# Graph Generation Code
#=======================================================================

def generateGraph():
    """Create MST from instance file"""
    with open(filename) as inputfile:
        results = list(csv.reader(inputfile))
    V = results.pop(0)[0]
    RB = results.pop(-1)[0]

    # build nodes from inputfile
    G = nx.Graph()
    for i in xrange(int(V)):
        node_color = RB[0].lower()
        RB = RB[1:] if len(RB) > 1 else RB
        G.add_node(i+1, color=node_color)

    # add edges
    for i in G.nodes():
        weights = results.pop(0)[0].split()
        for k, weight in enumerate(weights):
            G.add_edge(i, k+1, weight=int(weight))
    return G

def generatePath(G):
    even_v = []
    MST = nx.minimum_spanning_tree(G)
    for v in MST.nodes():
        if len(MST.neighbors(v)) % 2 == 0:
            even_v.append(v)
    O = G.copy()
    for v in even_v:
        O.remove_node(v)
    matching = []
    while len(O.edges()) > 1:
        minEdge = findMinEdge(O)
        O.remove_node(minEdge[0])
        O.remove_node(minEdge[1])
        matching.append(minEdge)
    MST = nx.MultiGraph(MST)
    MST.add_weighted_edges_from(matching)
    eulerTour = list(nx.eulerian_circuit(MST))
    MST = nx.Graph(MST)
    maxEdge = findMaxEdge(MST)
    rudrataPath = findEulerPath(maxEdge, eulerTour) #eulerPath
    swap = nx.Graph()
    swap.add_nodes_from(MST.nodes(data=True))
    swap.add_weighted_edges_from([(u, v, G[u][v]['weight']) for (u, v) in rudrataPath])
    if len(swap.nodes()) > 4:
        swap = double_edge_swap(G, swap, nswap=2000, max_tries=10000)
    path = edgesToPath(swap.edges())
    problems = pathCheck(G, path)
    if problems > 0:
        path = CHEAT(G)
    TSP = ''
    for v in path:
        TSP += str(v) + ' '
    return TSP[:-1] # gets rid of final space

def findMinEdge(O):
    minWeight = 101
    minEdge = None
    for (u, v) in O.edges():
        if u != v and O[u][v]['weight'] < minWeight:
            minWeight = O[u][v]['weight']
            minEdge = (u, v, minWeight)
    return minEdge

def findMaxEdge(O):
    maxWeight = -1
    maxEdge = None
    for (u,v) in O.edges():
        if u != v and O[u][v]['weight'] > maxWeight:
            maxWeight = O[u][v]['weight']
            maxEdge = (u,v)
    return maxEdge

def findEulerPath(maxEdge, eulerTour):
    """Takes in a eulerTour and a maxedge and deletes the max edge such that
        we return a euler Path. We then reorder the euler path.
    """
    path1, path2 = [], []
    split = False
    for (u, v) in eulerTour:
        skip = False
        if (u, v) == maxEdge or (v, u) == maxEdge:
            split = True
        if split and not skip:
            path2.append((u, v))
        if not split:
            path1.append((u, v))
    path = path2 + path1  # heaviest edge in cycle removed
    vertices = [] # ordering of vertices visited
    for (u, v) in path:
        if u not in vertices:
            vertices.append(u)
        if v not in vertices:
            vertices.append(v)
    finalPath = zip(vertices[:-1], vertices[1:])
    return finalPath

def edgesToPath(edges):
    """input edges forms a path
    """
    O = nx.Graph()
    for (u, v) in edges:
        O.add_node(u)
        O.add_node(v)
        O.add_edge(u, v)
    odd_v = []
    for v in O.nodes():
        if len(O.neighbors(v)) % 2 == 1:
            odd_v.append(v)
    start_v = odd_v[0]
    end_v = odd_v[1]
    path = [start_v]
    while start_v != end_v:
        next_v = O.neighbors(start_v)[0]
        O.remove_node(start_v)
        start_v = next_v
        path.append(next_v)
    return path

def double_edge_swap(G, O, nswap=1, max_tries=100):
    """O is a graph,
    nswap is number of swaps to perform,
    max_tries are the maximum number of attempts
    this returns a graph after double edge swaps
    """
    if not nx.is_connected(O):
        raise nx.NetworkXError("Graph not connected")
    if O.is_directed():
        raise nx.NetworkXError("double_edge_swap() not defined for directed graphs")
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(O) < 4:
        raise nx.NetworkXError("graph has less than 4 vertices")
    n=0
    currentCost = cost(O)
    swapcount=0
    keys, degrees=zip(*O.degree().items())
    cdf=nx.utils.cumulative_distribution(degrees)
    window = 1
    while swapcount < nswap:
        wcount=0
        swapped=[]
        W = O.copy()
        while wcount < window and swapcount < nswap:
            (ui,xi) = nx.utils.discrete_sequence(2,cdistribution=cdf)
            if ui==xi:
                continue #same source, skip
            u=keys[ui]
            x=keys[xi]
            v=random.choice([n for n in O[u] if type(n) == int])
            y=random.choice([n for n in O[x] if type(n) == int])
            if v == y:
                continue #same source, skip
            if (x not in O[u]) and (y not in O[v]): #don't create parallel edges
                O.add_edge(u,x,weight=G[u][x]['weight'])
                O.add_edge(v,y,weight=G[v][y]['weight'])
                O.remove_edge(u,v)
                O.remove_edge(x,y)
                swapped.append((u,v,x,y)) # not sure about this
                swapcount+=1
            if pathCheck(G, edgesToPath(O.edges())) == 0:
                if cost(O) > currentCost:
                    O = W
                currentCost = cost(O)
                break
            swapcount+=1
            wcount+=1
        if nx.is_connected(O):
            window+=1
        else:
            while swapped:
                #if not connected undo changes
                (u,v,x,y) = swapped.pop()
                O.add_edge(u,v, weight=G[u][v]['weight'])
                O.add_edge(x,y, weight=G[x][y]['weight'])
                O.remove_edge(u,x)
                O.remove_edge(v,y)
                swapcount-=1
            window = int(math.ceil(float(window)/2))
        if pathCheck(G, edgesToPath(O.edges())) >= pathCheck(G, edgesToPath(W.edges())):
            O = W
        if n >= max_tries:
            return W
        n+=1
    return O

def pathCheck(G, listV):
    count = 0
    problems = 0
    lastColor = None
    RBString = ''
    for v in listV:
        RBString += G.node[v]['color']
        if G.node[v]['color'] == lastColor:
            count += 1
            if count > 2:
                problems += 1
                count = 0
        else:
            count = 0
        lastColor = G.node[v]['color']
    # print RBString
    return problems

def cost(O):
    """returns cost of all edges in a graph"""
    total = 0
    for (u, v) in O.edges():
        total += O[u][v]['weight']
    return total

def CHEAT(G):
    """ does random bipartite traversal"""
    red_nodes = []
    blue_nodes = []
    for u in G.nodes():
        if G.node[u]['color'] == 'r':
            red_nodes.append(u)
        elif G.node[u]['color'] == 'b':
            blue_nodes.append(u)
    random.shuffle(red_nodes)
    random.shuffle(blue_nodes)
    cheating = []
    for i in range(len(red_nodes)):
        cheating.append(red_nodes.pop(0))
        cheating.append(blue_nodes.pop(0))
    return cheating

def main():
    import sys
    if len(sys.argv) == 2:
        global filename
        filename = sys.argv[1]
        if filename == 'all':
            outfile = 'answer.out'
            output = open(outfile, 'w')
            for i in range(1,496):
                filename = str(i) + '.in'
                print filename
                G = generateGraph()
                TSP = generatePath(G)
                if i == 495:
                    output.write(TSP)
                else:
                    output.write(TSP + '\n')
        else:
            G = generateGraph()
            TSP = generatePath(G)
            # output.write(TSP)
    else:
        print "Usage: christofides.py [filename|all]"

if __name__ == "__main__": main()