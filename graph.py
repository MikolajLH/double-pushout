import igraph as ig
from itertools import chain
from copy import deepcopy
from random import shuffle
import json

# Graph class acting as a wrapper around ig.graph
# Each vertex in a graph has two attributes:
# - Label - which is a string and do not have to be unique
# - Index - which is an int, this attribute is used as a primary key for a vertex
class Graph:

    def __init__(self, vertices_list : list[tuple[int,str]], edges_list : list[tuple[int,int]]) -> None:
        Indices, Labels = zip(*vertices_list)

        # if inner structure of the graph is changed, this mappings may no longer be valid
        self.map_i_to_I = dict( enumerate(Indices) )
        self.map_I_to_i = dict( (v,k) for k,v in self.map_i_to_I.items() )

        # creates ig.graph and adds Indices and Labels to it's vertices
        self.graph = ig.Graph(len(vertices_list),[(self.map_I_to_i[v1I],self.map_I_to_i[v2I]) for v1I, v2I in edges_list], directed = True)
        self.graph.vs['Label'] = Labels
        self.graph.vs['Index'] = Indices

    # shuffels internal order of vertices in ig.graph
    def shuffle(self):
        Es = [(e.source_vertex['Index'],e.target_vertex['Index']) for e in self.graph.es]
        Indices = [v['Index'] for v in self.graph.vs]
        Labels = [v['Label'] for v in self.graph.vs]
        Vs = list(zip(Indices,Labels))
        shuffle(Vs)
        Is,Ls = zip(*Vs)
        self.map_i_to_I = dict( enumerate(Is) )
        self.map_I_to_i = dict( (v,k) for k,v in self.map_i_to_I.items() )

        self.graph = ig.Graph(len(Vs), [(self.map_I_to_i[v1I],self.map_I_to_i[v2I]) for v1I, v2I in Es], directed = True)
        self.graph.vs['Label'] = Ls
        self.graph.vs['Index'] = Is

    def has_Index(self, I : int) -> bool:
        return len(self.graph.vs.select(Index= I)) != 0

    # mapping from I to i
    def get_i_from_I(self, I : int):
        return self.graph.vs.find(Index= I).index

    # mapping from i to I
    def get_I_from_i(self, i : int):
        return self.graph.vs[i]['Index']

    # convert graph to json string
    def to_json(self, indent= 4):
        d = {
            'Vertices' : [(I,L) for I,L in zip(self.graph.vs['Index'],self.graph.vs['Label'])],
            'Edges' : [(e.source_vertex['Index'],e.target_vertex['Index']) for e in self.graph.es]
        }

        return json.dumps(d,indent= indent)
    
    def to_dict(self):
        d = {
            'Vertices' : [(I,L) for I,L in zip(self.graph.vs['Index'],self.graph.vs['Label'])],
            'Edges' : [(e.source_vertex['Index'],e.target_vertex['Index']) for e in self.graph.es]
        }
        return d


def Graph_from_dict( d : dict):
    Vs = [(I,L) for I,L in d['Vertices']]
    Es = [(S,T) for S,T in d['Edges'] ]
    return Graph(Vs,Es)
    
# returns ([subgraph_vertices_indices],[subgraph_edges_indices])
# these indices are the the indices used by igrap.Graph object internaly,
# use G.graph.vs.select(subgraph_vertices_indices) in order to obtain sequence of igraph vertex objects that span the subgraph
# use G.graph.es.select(subgraph_edges_indices) in order to obtain sequence of igraph edge objects that are contained by subgraph
def get_subgraph(G : Graph, L : Graph):

    # finds a list of vertices that span subgraph L inside G
    potential_subgraphs_vertices = G.graph.get_subisomorphisms_vf2(L.graph,node_compat_fn= lambda G1,G2,v1,v2: G1.vs[v1]['Label'] == G2.vs[v2]['Label'])

    
    # if no such vertices where found return empty result
    if len(potential_subgraphs_vertices) == 0:
        return ([],[])

    psL = [ [G.graph.vs[i]['Label'] for i in sg] for sg in potential_subgraphs_vertices ]

    # replaces igraph inner index i with Index I in a list of potential subgraphs
    for k,sgvs in enumerate(potential_subgraphs_vertices):
        potential_subgraphs_vertices[k] = [G.map_i_to_I[i] for i in sgvs]
        
    # edges of graphs L and G in form of (source,target)
    L_edges = [(e.source_vertex['Index'],e.target_vertex['Index']) for e in L.graph.es]
    G_edges = [(e.source_vertex['Index'],e.target_vertex['Index']) for e in G.graph.es]

    
    maps_list = [
        dict( (k,v) for k,v in zip(sgI,[v['Index'] for v in L.graph.vs.select(lambda v: v['Label'] in sgL)]) )
        for sgI, sgL in zip(potential_subgraphs_vertices, psL) 
        ]

    # for ach potetnial subgraph vertices finds edges that are analogous to edges in Graph L 
    potential_subgraphs_edges = [[(s,t) for s,t in G_edges if s in sgvs and t in sgvs] for sgvs,map_f in zip(potential_subgraphs_vertices, maps_list)]

    # from potential subgraphs filters only graphs that have both - propper vertices and full set of propper edges
    subgraphs = list(filter(lambda sgvs_sges: len(sgvs_sges[1]) == len(L_edges),zip(potential_subgraphs_vertices,potential_subgraphs_edges)))

    # if no such subgraph exists returns empty result
    if len(subgraphs) == 0:
        return ([],[])

    # transforms output and returns first proper subgraph
    subgraphs = [([G.map_I_to_i[I] for I in sgvs],[G.graph.get_eid(G.map_I_to_i[s],G.map_I_to_i[t]) for s,t in sges]) for sgvs, sges in subgraphs]
    
    return subgraphs[0]
    

# returns new graph as a result of applying production (L,K,R) to graph G
def double_pushout(Gparam : Graph, L : Graph, K : Graph, R : Graph, Subgraph : tuple[list[int],list[int]]):
    G = deepcopy(Gparam)
    Subgraph_vi, Subgraph_ei = Subgraph
    Subgraph_Indices = [G.get_I_from_i(i) for i in Subgraph_vi]
    Subgraph_Labels = [v['Label'] for v in G.graph.vs.select(Subgraph_vi)]
    
    # creates mapping from G.Index to L.Index and its reverse
    map_GI_to_LI = dict( (k,v) for k,v in zip(Subgraph_Indices,[v['Index'] for v in L.graph.vs.select(lambda v: v['Label'] in Subgraph_Labels)]) )
    map_LI_to_GI = dict( (v,k) for k,v in map_GI_to_LI.items() )

    # creates mapping from G.Index to K.Index and its reverse 
    map_GI_to_KI = dict( (k,v) for k,v in map_GI_to_LI.items() if K.has_Index(v) )
    map_KI_to_GI = dict( (v,k) for k,v in map_GI_to_KI.items() )

    # saves attributes (Label and Index) of vertices from G graph that are analogous to K
    K_vertices_attributes = [G.graph.vs[G.map_I_to_i[map_KI_to_GI[I]]].attributes() for I in K.graph.vs['Index']]

    # saves edges that production should ignore
    saved_edges_seq = G.graph.es.select(
        lambda e:
        len(K.graph.es.select(
            lambda q: 
                e.source_vertex['Index'] == map_KI_to_GI[q.source_vertex['Index']] and 
                e.target_vertex['Index'] == map_KI_to_GI[q.target_vertex['Index']])
            ) > 0
        or
        e.source_vertex['Index'] in map_GI_to_KI.keys() and e.target not in Subgraph_vi # only source is contained by graph K and the target is a vertex not affected by production
        or
        e.source not in Subgraph_vi and e.target_vertex['Index'] in map_GI_to_KI.keys() # only target is contained by graph K and the source is a vertex not affected by production
        )
    

    saved_edges = [ (G.map_i_to_I[e.source],G.map_i_to_I[e.target]) for e in saved_edges_seq ]

    # delets found subgraph L so it can be replaced by R
    G.graph.delete_vertices(Subgraph_vi)

    # adds saved vertices and edges that should be ignored by production
    for att in K_vertices_attributes:
        G.graph.add_vertices(1,att)
    G.graph.add_edges( (G.get_i_from_I(s), G.get_i_from_I(t)) for s,t in saved_edges )

    new_GI = max(G.graph.vs['Index']) + 1
    for v in R.graph.vs:
        I = v['Index']
        if G.has_Index(I) and I not in K.graph.vs['Index']:
            v['Index'] = new_GI
            new_GI = new_GI + 1

    # finds vertices that only appear in graph R and not in graph K
    exclusive_R_vertices_attributes = [R.graph.vs[R.get_i_from_I(I)].attributes() for I in R.graph.vs['Index'] if I not in K.graph.vs['Index'] ]
    exclusive_R_vertices_Indices = [att['Index'] for att in exclusive_R_vertices_attributes]
    exclusive_R_vertices_Labels = [att['Label'] for att in exclusive_R_vertices_attributes]

    # these vertices will be added to graph G so we need to assign new unique Index I that will be added to graph G to each of them
    max_GI = max(G.graph.vs['Index'])
    # creates mapping from graph G Indices to graph R Indices
    map_GI_to_RI = dict( (k,v) for k,v in zip(range(max_GI + 1,max_GI + 1 + len(exclusive_R_vertices_Indices) + 1),exclusive_R_vertices_Indices))

    # adds R vertices to graph G, each vertex gets new unique G Index
    for new_GI, L in zip(map_GI_to_RI.keys(),exclusive_R_vertices_Labels):
        G.graph.add_vertices(1,dict(Index= new_GI,Label= L))

    # updates the mapping and creates it's reverse
    map_GI_to_RI = dict( chain( map_GI_to_RI.items(), map_GI_to_KI.items() ) )
    map_RI_to_GI = dict( (v,k) for k,v in map_GI_to_RI.items())


    # adds new edges introduced by graph R
    for e in R.graph.es:
        s = G.get_i_from_I(map_RI_to_GI[e.source_vertex['Index']])
        t = G.get_i_from_I(map_RI_to_GI[e.target_vertex['Index']])
        if G.graph.get_eid(s,t, error= False) == -1:
            G.graph.add_edge(s,t)
        

    # updates Indices in graph G
    G.graph.vs['Index'] = [ i + 1 for i in range(len(G.graph.vs))]

    # since the internal structure of ig.graph was changed we need to update these two mappings
    G.map_i_to_I = dict( enumerate(G.graph.vs['Index']) )
    G.map_I_to_i = dict( (v,k) for k,v in G.map_i_to_I.items() )

    return G

# generator that yields consecutive parts of animation
# when returns None double_pushout can't be applied
# when returns (G,None,None) animation has ended
# otherwise returns
# (
#   G,
#   (list of vertices to hide in graph G, list of edges to hide in graph G) ,
#   (list of vertices to emphasize in graph G, list of edges to emphasize in graph G)
# )
# both edges and vertices are represented by their internal index in ig.graph structure
def double_pushout_animation(Gparam : Graph, L : Graph, K : Graph, R : Graph, Subgraph : tuple[list[int],list[int]]):
    if len(Subgraph) == 0 or len(Subgraph[0]) == 0:
        yield None
    else:
        G = deepcopy(Gparam)
        Subgraph_vi, Subgraph_ei = Subgraph
        Subgraph_Indices = [G.get_I_from_i(i) for i in Subgraph_vi]
        Subgraph_Labels = [v['Label'] for v in G.graph.vs.select(Subgraph_vi)]

        ##########
        yield (G, ([], []), ([],[]) )
        ##########
    
        # creates mapping from G.Index to L.Index and its reverse
        map_GI_to_LI = dict( (k,v) for k,v in zip(Subgraph_Indices,[v['Index'] for v in L.graph.vs.select(lambda v: v['Label'] in Subgraph_Labels)]) )
        map_LI_to_GI = dict( (v,k) for k,v in map_GI_to_LI.items() )

        # creates mapping from G.Index to K.Index and its reverse 
        map_GI_to_KI = dict( (k,v) for k,v in map_GI_to_LI.items() if K.has_Index(v) )
        map_KI_to_GI = dict( (v,k) for k,v in map_GI_to_KI.items() )
    

        ##########
        yield (G, (Subgraph_vi, Subgraph_ei), ([],[]) )
        ##########
    
        # saves attributes (Label and Index) of vertices from G graph that are analogous to K
        K_vertices_attributes = [G.graph.vs[G.map_I_to_i[map_KI_to_GI[I]]].attributes() for I in K.graph.vs['Index']]

        # saves edges that production should ignore
        saved_edges_seq = G.graph.es.select(
            lambda e:
            len(K.graph.es.select(
                lambda q: 
                    e.source_vertex['Index'] == map_KI_to_GI[q.source_vertex['Index']] and 
                    e.target_vertex['Index'] == map_KI_to_GI[q.target_vertex['Index']])
                ) > 0
            or
            e.source_vertex['Index'] in map_GI_to_KI.keys() and e.target not in Subgraph_vi # only source is contained by graph K and the target is a vertex not affected by production
            or
            e.source not in Subgraph_vi and e.target_vertex['Index'] in map_GI_to_KI.keys() # only target is contained by graph K and the source is a vertex not affected by production
            )

        saved_edges = [ (G.map_i_to_I[e.source],G.map_i_to_I[e.target]) for e in saved_edges_seq ]

        ##########
        Kgraph_vi = [ G.map_I_to_i[map_KI_to_GI[I]] for I in K.graph.vs['Index'] ]
        outter_ei = [ e.index for e in saved_edges_seq ]
        yield (G, (Subgraph_vi, Subgraph_ei), (Kgraph_vi,outter_ei) )
        ##########

        G.graph.delete_vertices(Subgraph_vi)
    
        ##########
        #problematic since it can yield graph without vertices and that brakes drawing algorithm
        #yield (G, ([],[]), ([],[]) )
        ##########

        # adds saved vertices and edges that should be ignored by production
        for att in K_vertices_attributes:
            G.graph.add_vertices(1,att)
        G.graph.add_edges( (G.get_i_from_I(s), G.get_i_from_I(t)) for s,t in saved_edges )


        ##########
        Kgraph_vi = [ G.get_i_from_I(map_KI_to_GI[I]) for I in K.graph.vs['Index'] ]
        yield (G, ([],[]), (Kgraph_vi,[]) )
        ##########

        new_GI = max(G.graph.vs['Index']) + 1
        for v in R.graph.vs:
            I = v['Index']
            if G.has_Index(I) and I not in K.graph.vs['Index']:
                v['Index'] = new_GI
                new_GI = new_GI + 1
                
        # finds vertices that only appear in graph R and not in graph K
        exclusive_R_vertices_attributes = [R.graph.vs[R.get_i_from_I(I)].attributes() for I in R.graph.vs['Index'] if I not in K.graph.vs['Index'] ]
        exclusive_R_vertices_Indices = [att['Index'] for att in exclusive_R_vertices_attributes]
        exclusive_R_vertices_Labels = [att['Label'] for att in exclusive_R_vertices_attributes]

        # these vertices will be added to graph G so we need to assign new unique Index I that will be added to graph G to each of them
        max_GI = max(G.graph.vs['Index'])
        # creates mapping from graph G Indices to graph R Indices
        map_GI_to_RI = dict( (k,v) for k,v in zip(range(max_GI + 1,max_GI + 1 + len(exclusive_R_vertices_Indices) + 1),exclusive_R_vertices_Indices))

        # adds R vertices to graph G, each vertex gets new unique G Index
        for new_GI, L in zip(map_GI_to_RI.keys(),exclusive_R_vertices_Labels):
            G.graph.add_vertices(1,dict(Index= new_GI,Label= L))


        # updates the mapping and creates it's reverse
        map_GI_to_RI = dict( chain( map_GI_to_RI.items(), map_GI_to_KI.items() ) )
        map_RI_to_GI = dict( (v,k) for k,v in map_GI_to_RI.items())

        ##########
        Rgraph_vi = [ G.get_i_from_I(map_RI_to_GI[I]) for I in R.graph.vs['Index'] ]
        yield (G, ([],[]), (Rgraph_vi,[]) )
        ##########
    

        Rgraph_ei = []
        # adds new edges introduced by graph R
        for e in R.graph.es:
            s = G.get_i_from_I(map_RI_to_GI[e.source_vertex['Index']])
            t = G.get_i_from_I(map_RI_to_GI[e.target_vertex['Index']])
            if G.graph.get_eid(s,t, error= False) == -1:
                G.graph.add_edge(s,t)
            

        # updates Indices in graph G
        G.graph.vs['Index'] = [ i + 1 for i in range(len(G.graph.vs))]

        # since the internal structure of ig.graph was changed we need to update these two mappings
        G.map_i_to_I = dict( enumerate(G.graph.vs['Index']) )
        G.map_I_to_i = dict( (v,k) for k,v in G.map_i_to_I.items() )


        ##########
        yield (G, ([],[]), (Rgraph_vi,Rgraph_ei) )
        ##########

    
        ##########
        yield (G, None, None) 
