import matplotlib.pyplot as plt
from graph import Graph
from graph import double_pushout
from graph import get_subgraph
from graph import double_pushout_animation
from graph import Graph_from_dict
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import matplotlib.animation as animation
import igraph as ig
import json


# Window Layout
# Creating a window and spliting it into 3 horizontal stripes - top for production, middle for graph and bottom for control buttons
fig = plt.figure('double-pushout demo',layout="constrained")
subfigs = fig.subfigures(3, 1, wspace=0.07, height_ratios=[0.25,0.70,0.05])

# production stripe
production_axs = subfigs[0].subplots(1, 5, width_ratios=[0.05,0.30,0.30,0.30,0.05])

production_axs[1].set_title('L')
production_axs[2].set_title('K')
production_axs[3].set_title('R')

prev_production_button = Button(production_axs[0],'⇦')
next_production_button = Button(production_axs[4],'⇨')

# graph stripe
graph_ax = subfigs[1].subplots(1, 1)

# control buttons stripe
apply_production_button = Button(subfigs[2].add_axes([0.425,0.5,0.1,1.0]), 'apply')
animation_control_button = Button(subfigs[2].add_axes([0.525,0.5,0.05,1.0]), '▶')


undo_newest_production_button = Button(subfigs[2].add_axes([0,0.5,0.1,1.0]), 'undo')
reset_active_graph_button = Button(subfigs[2].add_axes([0.1,0.5,0.1,1.0]), 'reset')
redraw_active_graph_button = Button(subfigs[2].add_axes([0.2,0.5,0.1,1.0]), 'redraw')
inv_prod_button = Button(subfigs[2].add_axes([0.3,0.5,0.05,1.0]), 'inv')


command_line = TextBox(subfigs[2].add_axes([0.75,0.5,0.24,1.0]),'')

next_graph_button = Button(subfigs[2].add_axes([0.575,0.5,0.05,1.0]),'⇨')
prev_graph_button = Button(subfigs[2].add_axes([0.375,0.5,0.05,1.0]),'⇦')
#---------------

# predefined graphs
G1 = Graph([(i + 1,L) for i,L in enumerate(['C','E','D','H','F','G','I'])],[(3,1),(4,3),(3,2),(3,5),(5,2),(6,5),(4,6),(5,7)])
L1 = Graph([(1,'D'),(2,'E'),(3,'F')],[(1,3),(3,2),(1,2)])
K1 = Graph([(1,'D'),(3,'F')],[])
R1 = Graph([(1,'D'),(5,'N'),(4,'M'),(3,'F')],[(1,5),(5,4),(3,5),(4,3),(4,1)])

L2 = Graph([(1,'D')],[])
K2 = Graph([(1,'D')],[])
R2 = Graph([(1,'D'),(2,'Z')],[(1,2)])

L3 = Graph([(1,'Z'),(2,'Z')],[])
K3 = Graph([(1,'Z'),(2,'Z')],[])
R3 = Graph([(1,'Z'),(2,'Z')],[(1,2)])
#-----------------

# List of productions and list of graphs that will be used by program

# productions is a list of triplets; each triplet represent double-pushout production - (L,K,R)
productions = [(L1,K1,R1),(L2,K2,R2),(L3,K3,R3)]

# each graphs element is a list containing one Graph; when a production is applied this inner list appends new Graph resulting from applied production
# this way it is easy to undo production on a specific graph by popping last element of it's list.
graphs = [[G1],[L1]]
#-----------------


class Index:
    #current active production index
    production_ind = 0
    #current active graph list index to get current active graph use G = graphs[self.graph_ind][-1]
    graph_ind = 0

    gda_name = 'kk'
    # subgraph contains either two empty list when the current active production can't be applied to current active graph
    # or list of indices of vertices and list of indices of edges of found subgraph in current active graph
    subgraph = ([],[])

    
    animation = None
    animation_paused = True
    animation_interval = 1000
    

    # checks if the current active graph contains  subgraph of L graph from current active production, which is mandatory for a production to be applied to a graph
    def update_subgraph(self):
        L = productions[self.production_ind][0]
        G = graphs[self.graph_ind][-1]
        self.subgraph = get_subgraph(G,L)

    # increments current active production index, updates subgraph
    def next_production(self,event):
        self.production_ind = min(len(productions) - 1,self.production_ind + 1)
        self.update_subgraph()
        self.draw_production()
        self.draw_graph()
        
    # decrements current active production index, updates subgraph
    def prev_production(self,event):
        self.production_ind = max(0,self.production_ind - 1)
        self.update_subgraph()
        self.draw_production()
        self.draw_graph()

    # increments current active graph index, updates subgraph
    def next_graph(self,event):
        self.graph_ind = min(len(graphs) - 1,self.graph_ind + 1)
        self.update_subgraph()
        self.draw_graph()
    
    # decrements current active graph index, updates subgraph
    def prev_graph(self,event):
        self.graph_ind = max(0,self.graph_ind - 1)
        self.update_subgraph()
        self.draw_graph()
    
    # appends result of current active production on current active graph to current active graph list
    def apply_production(self,event):
        if len(self.subgraph) == 0 or len(self.subgraph[0]) == 0:
            return
        
        L,K,R = productions[self.production_ind]
        G = graphs[self.graph_ind][-1]

        Q = double_pushout(G,L,K,R,self.subgraph)

        graphs[self.graph_ind].append(Q)
        self.update_subgraph()
        self.draw_graph()
        
    # erases last element of current active graph list
    def undo_newest_production(self,event):
        if len(graphs[self.graph_ind]) != 1:
            graphs[self.graph_ind].pop()
            self.update_subgraph()
            self.draw_graph()
    
    # reset current active graph list by cutting the tail of the list leaving only the first element
    def reset_active_graph(self,event):
        if len(graphs[self.graph_ind]) != 1:
            graphs[self.graph_ind] = [graphs[self.graph_ind][0]]
            self.update_subgraph()
            self.draw_graph()
    
    # tries to redraw the current active graph by shuffling order of vertices in the inner structure of ig.graph
    def redraw_active_graph(self, event):
        if len(graphs[self.graph_ind]) != 0:
            graphs[self.graph_ind][-1].shuffle()
            self.update_subgraph()
            self.draw_graph()

    # redraws the production 
    def draw_production(self):
        L,K,R = productions[self.production_ind]
        production_axs[1].cla()
        production_axs[2].cla()
        production_axs[3].cla()

        def plot(G,ax):
            ig.plot(
                G.graph,
                target=ax,
                layout="kk",
                vertex_size=25.3,
                vertex_color='green',
                vertex_frame_width=4.0,
                vertex_frame_color="white",
                vertex_label=[str(a) + ' ' + str(b) for a,b in zip(G.graph.vs["Label"],G.graph.vs["Index"])],
                vertex_label_size=8.0,
                edge_width=1,
                edge_color= 'black'
            )
        plot(L,production_axs[1])
        plot(K,production_axs[2])
        plot(R,production_axs[3])
        production_axs[1].set_title('L')
        production_axs[2].set_title('K')
        production_axs[3].set_title('R')
        plt.draw()

    # redraws the current active graph and highlights the subgraph of current active production if it exists
    def draw_graph(self):
        G = graphs[self.graph_ind][-1]
        v_colors = [ 'purple' for _ in G.graph.vs ]
        e_colors = [ 'black' for _ in G.graph.es]
        Vs, Es = self.subgraph
        for v in Vs:
            v_colors[v] = 'red'
        for e in Es:
            e_colors[e] = 'orange'

        graph_ax.cla()
        ig.plot(
            G.graph,
            target=graph_ax,
            layout=self.gda_name,
            vertex_size=25.3,
            vertex_color=v_colors,
            vertex_frame_width=4.0,
            vertex_frame_color="white",
            vertex_label= [str(a) for a in G.graph.vs["Label"]],#[str(a) + ' ' + str(b) for a,b in zip(G.graph.vs["Label"],G.graph.vs["Index"])],
            vertex_label_size=7.0,
            edge_width=2,
            edge_color= e_colors
            )
        plt.draw()
    
    def submit(self, expr):
        if type(expr) != str:
            return
        expression = str(expr)
        command_line.set_val('')
        args = expression.strip().split()
        if len(args) == 0:
            return

        # delete current graph
        if args[0] == 'dg':
            if len(graphs) > 1:
                del graphs[self.graph_ind]
                self.graph_ind = max(0, self.graph_ind - 1)
                self.update_subgraph()
                self.draw_graph()
                return

        # delete current production
        if args[0] == 'dp':
            if len(productions) > 1:
                del productions[self.production_ind]
                self.production_ind = max(0, self.production_ind - 1)
                self.update_subgraph()
                self.draw_production()
                self.draw_graph()
                return

        # reverse current production
        if args[0] == 'rp':
            L,K,R = productions[self.production_ind]
            productions[self.production_ind] = (R,K,L)
            self.update_subgraph()
            self.draw_production()
            self.draw_graph()
        
        if args[0] == 'np':
            L,K,R = Graph([(1,'')],[]),Graph([(1,'')],[]),Graph([(1,'')],[])
            productions.append((L,K,R))
            self.update_subgraph()
            self.draw_production()
            self.draw_graph()
            return

        L,K,R = productions[self.production_ind]
        get_P = { 'L' : L, 'K' : K, 'R' : R }

        if len(args) == 2:
            # save graph as json file
            if args[0] == 'sg':
                self.save_graph(args[1])
            # load graph from json file
            if args[0] == 'lg':
                self.load_graph(args[1])
            # save production as json file
            if args[0] == 'sp':
                self.save_production(args[1])
            # load production from json file
            if args[0] == 'lp':
                self.load_production(args[1])
            # change drawing algorithm
            if args[0] == 'ca':
                self.change_graph_drawing_algorithm(args[1])
            # change interval in animation
            if args[0] == 'ci':
                self.animation_interval = int(args[1])
        elif len(args) == 3:
            if args[0] == 'rv':
                P = get_P[args[1]]
                I = int(args[2])
                if len(P.graph.vs) > 1:
                    for v in P.graph.vs(Index= I):
                        P.graph.delete_vertices(v.index)
        elif len(args) == 4:
            if args[0] == 'cL':
                P = get_P[args[1]]
                I = int(args[2])
                new_Label = args[3]
                for v in P.graph.vs(Index=I):
                    v['Label'] = new_Label
            if args[0] == 'cI':
                P = get_P[args[1]]
                I = int(args[2])
                new_I = int(args[3])
                if len(P.graph.vs(Index=new_I)) == 0:
                    for v in P.graph.vs(Index=I):
                        v['Index'] = new_I
            if args[0] == 'av':
                P = get_P[args[1]]
                new_I = int(args[2])
                new_Label = args[3]
                if len(P.graph.vs(Index= new_I)) == 0:
                    P.graph.add_vertices(1,dict(Index= new_I,Label= new_Label))
            if args[0] == 'ae':
                P = get_P[args[1]]
                sI = int(args[2])
                tI = int(args[3])
                if sI != tI:
                    for si,ti in zip(P.graph.vs(Index=sI), P.graph.vs(Index=tI)):
                        P.graph.add_edge(si,ti)
            if args[0] == 're':
                P = get_P[args[1]]
                sI = int(args[2])
                tI = int(args[3])
                
                ss = P.graph.vs(Index= sI)
                ts = P.graph.vs(Index= tI)

                if len(ss) != 0 and len(ss) != 0:
                    eid = P.graph.get_eid(ss[0].index,ts[0].index, error= False)
                    if eid != -1:
                        P.graph.delete_edges(eid)
                    
            self.update_subgraph()
            self.draw_production()
            self.draw_graph()

    def change_graph_drawing_algorithm(self, name : str):
        if name in ('circle', 'circular', 'drl', 'kk', 'rt', 'tree'):
            self.gda_name = name

    def save_graph(self, path : str):
        path = path.split('.')[0] + '.json'
        G = graphs[self.graph_ind][-1]
        with open(path, 'w') as out:
            out.write(G.to_json())
        print('graph saved')

    def load_graph(self, path : str):
        path = path.split('.')[0] + '.json'
        with open(path, 'r') as input_file:
            json_dict = json.load(input_file)
        Vs = [(I,L) for I,L in json_dict['Vertices']]
        Es = [(S,T) for S,T in json_dict['Edges'] ]
        graphs.append([Graph(Vs,Es)])
        print('graph loaded')

    def save_production(self, path : str):
        L,K,R = productions[self.production_ind]
        path = path.split('.')[0] + '.json'
        Prod_dict = { 'L' : L.to_dict(), 'K' : K.to_dict(), 'R' : R.to_dict() }
        with open(path,'w') as out:
            json.dump(Prod_dict,out, indent= 4)
        print('production saved')

    def load_production(self, path : str):
        path = path.split('.')[0] + '.json'
        with open(path, 'r') as input_file:
            json_dict = json.load(input_file)

        L = Graph_from_dict(json_dict['L'])
        K = Graph_from_dict(json_dict['K'])
        R = Graph_from_dict(json_dict['R'])

        productions.append((L,K,R))
        print('production loaded')

    def production_animation(self, frame):
        if frame is None:
            self.animation = None
            animation_control_button.label.set_text('▶')
            self.animation_paused = True
            return

        G, inviG, emphG = frame
        if inviG is None or emphG is None:
            self.animation = None
            graphs[self.graph_ind].append(G)
            animation_control_button.label.set_text('▶')
            self.animation_paused = True
            self.update_subgraph()
            self.draw_graph()
            return

        graph_ax.clear()
        inviVs, inviEs = inviG
        emphVs, emphEs = emphG

        v_colors = [ 'cyan' for _ in G.graph.vs ]
        e_colors = [ 'gray' for _ in G.graph.es ]

        visible = [ True for _ in G.graph.vs ]
        for v in inviVs:
            v_colors[v] = 'white'
            visible[v] = False
        for e in inviEs:
            e_colors[e] = 'white'
        for v in emphVs:
            v_colors[v] = 'green'
            visible[v] = True
        for e in emphEs:
            e_colors[e] = 'pink'

        ig.plot(
            G.graph,
            target= graph_ax,
            layout= self.gda_name,
            vertex_size= 0.3,
            vertex_color= v_colors,
            vertex_frame_width= 4.0,
            vertex_frame_color= "white",
            vertex_label_size=7.0,
            vertex_label= [ str(I) if visible[v] else '' for v,I in enumerate(G.graph.vs["Label"]) ],
            edge_width=2,
            edge_color= e_colors
            )
        return graph_ax.get_children()[:-1]

    def apply_with_animation(self, event):
        def helper():
            G = graphs[self.graph_ind][-1]
            L,K,R = productions[self.production_ind]
            def wrapper():
                for e in double_pushout_animation(G,L,K,R, self.subgraph):
                    yield e
            return wrapper
        def init_animation():
            return

        if self.animation is None:
            self.animation = animation.FuncAnimation(
                fig, self.production_animation, helper(),
                init_func= init_animation, interval=self.animation_interval, repeat= False )
            animation_control_button.label.set_text('❚❚')
            self.animation_paused = False
        else:
            if self.animation_paused:
                animation_control_button.label.set_text('❚❚')
                self.animation.resume()
                self.animation_paused = False
            else:
                animation_control_button.label.set_text('▶')
                self.animation.pause()
                self.animation_paused = True
        plt.draw()
                
    def inv_prod(self, event):
        L,K,R = productions[self.production_ind]
        productions[self.production_ind] = (R,K,L)
        self.update_subgraph()
        self.draw_production()
        self.draw_graph()
            
# assigning callback functions to buttons
callback = Index()

next_production_button.on_clicked(callback.next_production)
prev_production_button.on_clicked(callback.prev_production)

next_graph_button.on_clicked(callback.next_graph)
prev_graph_button.on_clicked(callback.prev_graph)

apply_production_button.on_clicked(callback.apply_production)

undo_newest_production_button.on_clicked(callback.undo_newest_production)
reset_active_graph_button.on_clicked(callback.reset_active_graph)
redraw_active_graph_button.on_clicked(callback.redraw_active_graph)
animation_control_button.on_clicked(callback.apply_with_animation)

command_line.on_submit(callback.submit)
inv_prod_button.on_clicked(callback.inv_prod)

# draws scene and starts program
callback.prev_production(None)
plt.show()