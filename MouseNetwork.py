import numpy as np
import networkx as nx

class MouseNetwork():

    def __init__(self, mouse_dict):

        self.number_of_cells = len(mouse_dict["cell_types"])
        self.cell_types = mouse_dict["cell_types"]
        self.T = 2*len(np.squeeze(mouse_dict["spike_trains"][0])) + 1
    
    def find_connectivity(self, result_peaks, edge_dict):

        nodes = range(self.number_of_cells)
        G = nx.Graph()
        G_directed = nx.DiGraph()
        
        G.add_nodes_from(nodes)
        G_directed.add_nodes_from(nodes)


        edges = []
        edge_weights = []
        edge_weights2 = {}
        
        for pair in result_peaks.keys(): 
            if result_peaks[pair][0]:

                node1 = pair[0]; node2 = pair[1];
                node1_s = str(pair[0]); node2_s = str(pair[1])
               
                G.add_edge(node1, node2)

                #print(result_peaks[pair][0])
                t1 = result_peaks[pair][0][0];
            
                #print("t1", t1)
                if (t1<self.T//2):
                    #print("t1<T//2")
                    G_directed.add_weighted_edges_from([(node2, node1, result_peaks[pair][1][0])])
                    edges.append([node2,node1])
                    edge_weights.append(result_peaks[pair][1][0])
                    try:
                        edge_weights2[(node2, node1)] = edge_dict[(node2_s, node1_s)]
                    except:
                        edge_weights2[(node2, node1)] = edge_dict[(node1_s, node2_s)]
                else:
                    #print("t1>self.T//2")
                    G_directed.add_weighted_edges_from([(node1, node2, result_peaks[pair][1][0])])
                    edges.append([node1,node2])
                    edge_weights.append(result_peaks[pair][1][0])
                    try:
                        edge_weights2[(node1, node2)] = edge_dict[(node1_s, node2_s)]
                    except:
                        edge_weights2[(node1, node2)] = edge_dict[(node2_s, node1_s)]

                    
                if len(result_peaks[pair][1])==2:
                    val1 = result_peaks[pair][1][0]
                    val2 = result_peaks[pair][1][1]
                    t2 = result_peaks[pair][0][1]
                    #print("val1, val2, t2", val1, val2, t2)
                    
                    if val2*5>val1:
                        if t2<self.T//2:
                            #print("another one 1")
                            G_directed.add_weighted_edges_from([(node2, node1, val2)])
                            edges.append([node2,node1])
                            edge_weights.append(val2)
                            try:
                                edge_weights2[(node2, node1)] = edge_dict[(node2_s, node1_s)]
                            except:
                                edge_weights2[(node2, node1)] = edge_dict[(node1_s, node2_s)]
                        else:
                            #print("another one 2")
                            G_directed.add_weighted_edges_from([(node1, node2, val2)])
                            edges.append([node1,node2])
                            edge_weights.append(val2)
                            try:
                                edge_weights2[(node1, node2)] = edge_dict[(node1_s, node2_s)]
                            except:
                                edge_weights2[(node1, node2)] = edge_dict[(node2_s, node1_s)]

        self.graph = G
        self.directed_graph = G_directed

        self.edges = np.array(edges)
        self.edge_weights = np.array(edge_weights)
        self.edge_weights2 = edge_weights2
        