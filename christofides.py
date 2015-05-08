from __future__ import division
import csv
import time
import networkx as nx

"""
Implementations of d-Heaps and Prim's MST following Tarjan. Includes testing
and visualization code for both.
"""

#=======================================================================
# Union-Find
#=======================================================================

class ArrayUnionFind:
    """Holds the three "arrays" for union find"""
    def __init__(self, S):
        self.group = dict((s,s) for s in S) # group[s] = id of its set
        self.size = dict((s,1) for s in S) # size[s] = size of set s
        self.items = dict((s,[s]) for s in S) # item[s] = list of items in set s

def make_union_find(S):
    """Create a union-find data structure"""
    return ArrayUnionFind(S)

def find(UF, s):
    """Return the id for the group containing s"""
    return UF.group[s]

def union(UF, a,b):
    """Union the two sets a and b"""
    assert a in UF.items and b in UF.items
    # make a be the smaller set
    if UF.size[a] > UF.size[b]:
        a,b = b,a
    # put the items in a into the larger set b
    for s in UF.items[a]:
        UF.group[s] = b
        UF.items[b].append(s)
    # the new size of b is increased by the size of a
    UF.size[b] += UF.size[a]
    # remove the set a (to save memory)
    del UF.size[a]
    del UF.items[a]

#=======================================================================
# Kruskal MST
#=======================================================================

def kruskal_mst(G):
    """Return a minimum spanning tree using kruskal's algorithm"""
    # sort the list of edges in G by their weight
    Edges = [(u, v, G[u][v]['weight']) for u,v in G.edges()]
    Edges.sort(cmp=lambda x,y: cmp(x[2],y[2]))

    UF = make_union_find(G.nodes())  # union-find data structure

    # for edges in increasing weight
    mst = [] # list of edges in the mst
    for u,v,d in Edges:
        setu = find(UF, u)
        setv = find(UF, v)
        # if u,v are in different components
        if setu != setv:
            mst.append((u,v))
            union(UF, setu, setv)
            snapshot_kruskal(G, mst)
    return mst

#=======================================================================
# MST Testing and Visualization Code
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

def draw_mst_graph(G, T={}, outfile=None):
    """Draw the MST graph, highlight edges given by the MST parent dictionary
    T. T should be in the same format as returned by prim_mst()."""

    import matplotlib.pyplot as plt

    # construct the attributes for the edges
    labels = dict((u,str(u)) for u in G.nodes())
    mst_edges = []
    adjacency_list = {}
    red_nodes = []
    blue_nodes = []
    pos = nx.circular_layout(G)
    for u in G.nodes():
        adjacency_list[u] = []
        if G.node[u]['color'] == 'r':
            red_nodes.append(u)
        elif G.node[u]['color'] == 'b':
            blue_nodes.append(u)

    for u,v in G.edges():
        if isinstance(T, dict):
            inmst = (u in T and v == T[u]) or (v in T and u == T[v])
        elif isinstance(T, nx.Graph):
            inmst = T.has_edge(u,v)
        if inmst:
            mst_edges.append((u, v))
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)

    # initialize plot
    plt.figure(facecolor="w", dpi=80)
    plt.margins(0,0)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    # draw nodes
    nx.draw_networkx_nodes(G,pos,
        nodelist=red_nodes,
        node_color='r',
        node_size=500,
        alpha=0.8)
    nx.draw_networkx_nodes(G,pos,
        nodelist=blue_nodes,
        node_color='b',
        node_size=500,
        alpha=0.8)

    # draw edges
    nx.draw_networkx_edges(G,pos,
        width=0.5,alpha=0.5)
    print 'mst_edges: ' + str(mst_edges)
    nx.draw_networkx_edges(G,pos,
        edgelist=mst_edges,
        width=2,alpha=0.5,edge_color='b')

    # draw labels
    nx.draw_networkx_labels(G,pos,labels)

    if outfile is not None:
        plt.savefig(outfile, format="pdf", dpi=150, bbox_inches='tight', pad_inches=0.0)
        plt.close()
    else:
        total_cost = 0
        for u, v in mst_edges:
            total_cost += G.edge[u][v]['weight']
        print "Time: %.3fs, Total Cost: %d, MST Edges: %s" % (time.time() - t, total_cost, mst_edges)
        plt.show()

def snapshot_kruskal(G, edges, pdf=True):
    T = nx.Graph()
    for u,v in edges: T.add_edge(u,v)
    draw_mst_graph(G, T, pdffile if pdf else None)

def test_kruskal():
    """Draw the MST for a random graph."""
    global pdffile, t
    pdffile = start_pdf("kruskal.pdf")
    t = time.time()
    N = generateGraph()
    snapshot_kruskal(N, kruskal_mst(N), False)
    close_pdf(pdffile)

def start_pdf(outfile):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(outfile)
    return pp

def close_pdf(pp):
    pp.close()

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
    path1 = []
    path2 = []
    weightList = [MST[u][v][0]['weight'] for (u,v) in eulerTour]
    k=1
    check = False
    for i, (u, v) in enumerate(eulerTour):
        k+=1
        if MST[u][v][0]["weight"] == max(weightList):
            k=0
            check = True
        if check and k!=0:
            path2.append((u,v))
        if not check:
            path1.append((u,v))
    path = path2 + path1
    vSet = []
    TSP = ''
    for (u, v) in path:
        if u not in vSet:
            TSP += str(u) + ' '
            vSet.append(u)
        if v not in vSet:
            TSP += str(v) + ' '
            vSet.append(v)
    return TSP[:-1] # gets rid of final space

def findMinEdge(O):
    minWeight = 100
    minEdge = None
    for (u, v) in O.edges():
        if u != v and O[u][v]['weight'] < minWeight:
            minWeight = O[u][v]['weight']
            minEdge = (u, v, minWeight)
    return minEdge

def findMaxEdge(O):
    maxWeight = 0
    maxEdge = None
    for (u,v) in O.edges():
        if u != v and O[u][v]['weight'] > maxWeight:
            maxWeight = O[u][v]['weight']
            maxEdge = (u,v, maxWeight)
    return maxEdge


def pathCheck(G, listV):
    # colorSet = []
    count = 0
    lastColor = None
    for v in listV:
        # colorSet.append(G[v]['color'])
        lastColor = G[v]['color']
        if count > 3:
            return False
        if G[v]['color'] == lastColor:
            count++
        else:
            count = 0
    return True


def main():
    import sys
    if len(sys.argv) == 2:
        global filename
        filename = sys.argv[1]
        outfile = 'answer.out'
        if filename == 'all':
            for i in range(1,496):
                filename = 'instances/' + str(i) + '.in'
                print filename
                G = generateGraph()
                TSP = generatePath(G)
                output = open(outfile, 'w')
                if i == 495:
                    output.write(TSP)
                else:
                    output.write(TSP + '\n')
        else:
            G = generateGraph()
            TSP = generatePath(G)
            output = open(outfile, 'w')
            output.write(TSP)
    else:
        print "Usage: christofides.py [filename|all]"

if __name__ == "__main__": main()
