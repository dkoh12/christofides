from __future__ import division
import time
import math
import random
import networkx as nx
from collections import defaultdict

"""
Implementations of d-Heaps and Prim's MST following Tarjan. Includes testing
and visualization code for both.
"""

ARITY = 3  # the branching factor of the d-Heaps

#=======================================================================
# d-Heap
#=======================================================================

class HeapItem(object):
    """Represents an item in the heap"""
    def __init__(self, key, item):
        self.key = key
        self.item = item
        self.pos = None

def makeheap(S):
    """Create a heap from set S, which should be a list of pairs (key, item)."""
    heap = list(HeapItem(k,i) for k,i in S)
    for pos in xrange(len(heap)-1, -1, -1):
        siftdown(heap[pos], pos, heap)
    return heap

def findmin(heap):
    """Return element with smallest key, or None if heap is empty"""
    return heap[0] if len(heap) > 0 else None

def deletemin(heap):
    """Delete the smallest item"""
    if len(heap) == 0: return None
    i = heap[0]
    last = heap[-1]
    del heap[-1]
    if len(heap) > 0:
        siftdown(last, 0, heap)
    return i

def heapinsert(key, item, heap):
    """Insert an item into the heap"""
    heap.append(None)
    hi = HeapItem(key,item)
    siftup(hi, len(heap)-1, heap)
    return hi

def heap_decreasekey(hi, newkey, heap):
    """Decrease the key of hi to newkey"""
    hi.key = newkey
    siftup(hi, hi.pos, heap)

def siftup(hi, pos, heap):
    """Move hi up in heap until its parent is smaller than hi.key"""
    p = parent(pos)
    while p is not None and heap[p].key > hi.key:
        heap[pos] = heap[p]
        heap[pos].pos = pos
        pos = p
        p = parent(p)
    heap[pos] = hi
    hi.pos = pos

def siftdown(hi, pos, heap):
    """Move hi down in heap until its smallest child is bigger than hi's key"""
    c = minchild(pos, heap)
    while c != None and heap[c].key < hi.key:
        heap[pos] = heap[c]
        heap[pos].pos = pos
        pos = c
        c = minchild(c, heap)
    heap[pos] = hi
    hi.pos = pos

def parent(pos):
    """Return the position of the parent of pos"""
    if pos == 0: return None
    return int(math.ceil(pos / ARITY) - 1)

def children(pos, heap):
    """Return a list of children of pos"""
    return xrange(ARITY * pos + 1, min(ARITY * (pos + 1) + 1, len(heap)))

def minchild(pos, heap):
    """Return the child of pos with the smallest key"""
    minpos = minkey = None
    for c in children(pos, heap):
        if minkey == None or heap[c].key < minkey:
            minkey, minpos = heap[c].key, c
    return minpos


#=======================================================================
# Heap Testing and Visualization Code
#=======================================================================

def bfs_tree_layout(G, root, rowheight = 0.02, nodeskip = 0.6):
    """Return node position dictionary, layingout the graph in BFS order."""
    def width(T, u, W):
        """Returns the width of the subtree of T rooted at u; returns in W the
        width of every node under u"""
        W[u] = sum(width(T, c, W)
            for c in T.successors(u)) if len(T.successors(u))>0 else 1.0
        return W[u]

    T = nx.bfs_tree(G, root)
    W = {}
    width(T, root, W)

    pos = {}
    left = {}
    queue = [root]
    while len(queue):
        c = queue[0]
        del queue[0]  # pop

        left[c] = 0.0  # amt of child space used up

        # posn is computed relative to the parent
        if c == root:
            pos[c] = (0,0)
        else:
            p = T.predecessors(c)[0]
            pos[c] = (
                pos[p][0] - W[p]*nodeskip/2.0 + left[p] + W[c]*nodeskip/2.0,
                pos[p][1] - rowheight
            )
            left[p] += W[c]*nodeskip

        # add the children to the queue
        for i,u in enumerate(G.successors(c)):
            queue.append(u)
    return pos

def snapshot_heap(heap):
    draw_heap(heap, pdffile)

def draw_heap(heap, outfile=None):
    """Draw the heap using matplotlib and networkx"""
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    for i in xrange(1, len(heap)):
        G.add_edge(parent(i), i)

    labels = dict((u, "%d" % (heap[u].key)) for u in G.nodes())

    plt.figure(facecolor="w", dpi=80)
    plt.margins(0,0)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    nx.draw_networkx(G,
        labels=labels,
        node_size = 700,
        node_color = "white",
        pos=bfs_tree_layout(G, 0))
    if outfile is not None:
        plt.savefig(outfile, format="pdf", dpi=150, bbox_inches='tight', pad_inches=0.0)
        plt.close()
    else:
        plt.show()

def test_heap():
    """Generate a random heap"""
    global pdffile
    pdffile = start_pdf("mst.pdf")
    draw_heap(makeheap((random.randint(0,100), 'a') for i in xrange(40)))


#=======================================================================
# Prim's minimum spanning tree algorithm
#=======================================================================

def prim_mst(G):
    """Compute the minimum spanning tree of G. Assumes each edge has an
    attribute 'weight' giving its weight. Returns a dictionary P such
    that P[u] gives the parent of u in the MST."""

    for u in G.nodes():
        G.node[u]['distto'] = float("inf")  # key stores the Prim key
        G.node[u]['heap'] = None         # heap = pointer to node's HeapItem
    parent = {}

    heap = makeheap([])
    v = G.nodes()[0]

    # go through vertices in order of closest to current tree
    while v != None:
        G.node[v]['distto'] = float("-inf") # v now in the tree

        snapshot_mst(G, parent)

        # update the estimated distance to each of v's neighbors
        for w in G.neighbors(v):
            # if new weight is smaller that old weight, update
            if G[v][w]['weight'] < G.node[w]['distto']:
                # closest tree node to w is v
                G.node[w]['distto'] = G[v][w]['weight']
                parent[w] = v

                # add to heap or decreae key if already in heap
                hi = G.node[w]['heap']
                if hi is None:
                    G.node[w]['heap'] = heapinsert(G.node[w]['distto'], w, heap)
                else:
                    heap_decreasekey(hi, G.node[w]['distto'], heap)
        # get the next vertex closest to the tree
        v = deletemin(heap)
        v = v.item if v is not None else None
    return parent

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

def generate_graph():
    """Create MST from instance file"""
    import csv
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

def snapshot_mst(G, parent):
    tree = dict((u,parent[u])
        for u in G.nodes()
            if G.node[u]['distto'] == float("-inf") and u in parent)
    draw_mst_graph(G, tree, pdffile)

def test_mst():
    """Draw the MST for a random graph."""
    global pdffile, t
    pdffile = start_pdf("mst.pdf")
    t = time.time()
    N = generate_graph()
    draw_mst_graph(N, prim_mst(N))
    close_pdf(pdffile)

def test_kruskal():
    """Draw the MST for a random graph."""
    global pdffile, t
    pdffile = start_pdf("kruskal.pdf")
    t = time.time()
    N = generate_graph()
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

    # maxEdge = findMaxEdge(MST)
    # print maxEdge

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

    print path1, path2
    path = path2 + path1

        # if u not in path:
        #     path.append(u)
        # if v not in path:
        #     path.append(v)
        # print (u,v)


    return path

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

def main():
    import sys
    if len(sys.argv) >= 1:
        global filename
        filename = sys.argv[1]
        G = generate_graph()
        print generatePath(G)
        # if sys.argv[1] == "heap": test_heap()
        # if sys.argv[1] == "mst": test_mst()
        # if sys.argv[1] == "kruskal": test_kruskal()
    else:
        print "Usage: greedy.py [heap|mst|kruskal]"

if __name__ == "__main__": main()
