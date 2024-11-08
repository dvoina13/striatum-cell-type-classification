from MouseNetwork import MouseNetwork
import pickle
import numpy as np
import torch
import networkx as nx
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data_spike_trains_cells_speed():
    
            mouse_list = ['642481', '648843', '642481', '642480', '642478', '655571', '642480', '655568', '642478', '655568', '666721', '661398', '648845', '661398', '655572', '655565', '666721']
            all_nwb_paths = ['ecephys_642481_2023-01-24_13-18-13_nwb',
             'ecephys_648843_2023-02-22_13-55-39_nwb',
             'ecephys_642481_2023-01-25_11-33-13_nwb',
             'ecephys_642480_2023-01-26_17-06-40_nwb',
             'ecephys_642478_2023-01-17_14-38-38_nwb',
             'ecephys_655571_2023-05-09_13-53-48_nwb',
             'ecephys_642480_2023-01-25_12-20-15_nwb',
             'ecephys_655568_2023-05-03_15-21-12_nwb',
             'ecephys_642478_2023-01-11_11-02-09_nwb',
             'ecephys_655568_2023-05-01_15-26-47_nwb',
             'ecephys_666721_2023-05-12_16-15-36_nwb',
             'ecephys_661398_2023-03-31_17-01-09_nwb',
             'ecephys_648845_2023-02-23_14-27-33_nwb',
             'ecephys_661398_2023-04-03_15-47-29_nwb',
             'ecephys_655572_2023-05-09_15-03-29_nwb',
             'ecephys_655565_2023-03-31_14-47-36_nwb',
             'ecephys_666721_2023-05-09_11-01-03_nwb']
            
            
            with open('saved_dictionary.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
            
            mice_ids = np.arange(len(loaded_dict))
            spike_trains = []
            cell_types = []
            
            for ind in range(len(mouse_list)):
                
                spike_trains.append(np.squeeze(loaded_dict[ind]["spike_trains"]))
                cell_types += loaded_dict[ind]["cell_types"]
            
            spike_trains = np.vstack(spike_trains)
            
            
            spike_trains_ = []
            running_speeds_ = []
            spike_trains_permuted = []
            cell_types = []
            loaded_average_spikes = []
            experiments = []
            interval = 10000
            
            for ind in range(len(mouse_list)):
            
                print("MOUSE", ind)
                cell_types_ = loaded_dict[ind]["cell_types"]
                for j in range(len(cell_types_)):
                    if cell_types_[j] == "D1":
                        cell_types.append(0)
                    elif cell_types_[j] == "D2":
                        cell_types.append(1)
                    else:    
                        cell_types.append(-10)
            
                experiments += [ind]*len(cell_types_);
                loaded_spikes = np.squeeze(loaded_dict[ind]["spike_trains"])
                running_speed = np.squeeze(loaded_dict[ind]["running_speed"])
                
                print(loaded_spikes.shape)
                loaded_spikes_coarse = []
                running_speed_coarse = []
                for i in range(int(loaded_spikes.shape[1]/interval)):
                    loaded_spikes_coarse.append(loaded_spikes[:,i*interval:(i+1)*interval].sum(1))
                    running_speed_coarse.append(np.ones(len(cell_types_))*running_speed[i*interval:(i+1)*interval].sum())
                    
                loaded_spikes_coarse = np.array(loaded_spikes_coarse)
                loaded_spikes_coarse = np.transpose(loaded_spikes_coarse, (1,0)).squeeze()
                running_speed_coarse = np.array(running_speed_coarse)
                running_speed_coarse = np.transpose(running_speed_coarse, (1,0)).squeeze()
                
                #create partitions
                loaded_spikes_coarse_permuted = []
                loaded_average_spikes_ = []
                num_div = 6115
                T_len = np.array(loaded_spikes_coarse).shape[1]; div = int(T_len//num_div)
                print(T_len, num_div, div)
                partitions = [div*k for k in range(num_div)]
                for j in range(len(cell_types_)):
                        loaded_spikes_coarse[j,:] = loaded_spikes_coarse[j,:]#/loaded_spikes_coarse.max()#(loaded_spikes_coarse[j,:] - loaded_spikes_coarse[j, :].mean())/loaded_spikes_coarse[j, :].std()**2
            
                        spike_count = []
                        for spikes in range(54):
                            spike_count.append(len(np.where(loaded_spikes_coarse[j,:] == spikes)[0]))
            
                        loaded_average_spikes_.append(np.array(spike_count))
                    
                        partitions = np.random.permutation(partitions)
                        spikes_coarse_temp = []
                        
                        if div == 1:
                            spikes_coarse_temp = loaded_spikes_coarse[j, partitions]
                        else:
                            for p in partitions:
                                spikes_coarse_temp += list(loaded_spikes_coarse[j, p:p+div])
                        
                        spikes_coarse_temp = np.array(spikes_coarse_temp)
                        loaded_spikes_coarse_permuted.append(spikes_coarse_temp)
                
                loaded_average_spikes.append(loaded_average_spikes_)    
                loaded_spikes_coarse_permuted = np.array(loaded_spikes_coarse_permuted)
            
                print("loaded_spikes_coarse", loaded_spikes_coarse.shape)
                running_speeds_.append(running_speed_coarse)
                spike_trains_.append(loaded_spikes_coarse)
                spike_trains_permuted.append(loaded_spikes_coarse_permuted)
            
            cell_types = np.array(cell_types)
            spike_trains_ = np.vstack(spike_trains_)
            running_speeds_ = np.vstack(running_speeds_)
            
            spike_trains_permuted = np.vstack(spike_trains_permuted)
            loaded_average_spikes = np.vstack(loaded_average_spikes)
            experiments = np.array(experiments)


            return spike_trains_, cell_types, running_speeds_, spike_trains_permuted, loaded_average_spikes, experiments, loaded_dict, all_nwb_paths


def load_graph(all_nwb_paths, loaded_dict, cell_types):

        i = 0
        Graph_all = []
        Directed_Graph_all = [];
        edge_weights = [];
        edge_weights2 = []
        indices_for_new_session = []
        ind_0 = []; ind_1 = [];
        mice = [];
        for ind, f in enumerate(all_nwb_paths):
            print(ind)
        
            file1 = "/Users/dorisvoina/Desktop/work_stuff/P3_GNNs/code_ocean/simple_correlations/result_peak_mouse_" + f[8:-4] + ".npy"
            print(file1)
            result_peaks = np.load(file1, allow_pickle=True)
            result_peaks = result_peaks.item()
        
            file2 = "/Users/dorisvoina/Desktop/work_stuff/P3_GNNs/code_ocean/all_edges_ecephys_" + f[8:-4] + "_nwb.npy"
            edge_dict = np.load(file2, allow_pickle=True)
            
            mouse = MouseNetwork(loaded_dict[ind])
            print("number of cells", mouse.number_of_cells)
            mouse.find_connectivity(result_peaks, edge_dict)
            mice.append(mouse)
        
            mapping = {}
            for j in range(len(mouse.graph.nodes)): mapping[list(mouse.graph.nodes)[j]] = j+i
            graph_int = nx.relabel_nodes(mouse.graph, mapping, copy=True)
            edges = np.array([list(list(graph_int.edges)[k]) for k in range(len(list(graph_int.edges)))])
            Graph_all.append(edges)
            
            graph_int_dir = nx.relabel_nodes(mouse.directed_graph, mapping, copy=True)
            directed_edges = np.array([list(list(graph_int_dir.edges)[k]) for k in range(len(list(graph_int_dir.edges)))])
            Directed_Graph_all.append(directed_edges)
        
            for pair in directed_edges:
                edge_info = mouse.directed_graph.get_edge_data(pair[0]-i, pair[1]-i, default=None)
                edge_weights.append(edge_info["weight"])
                edge_weights2.append(mouse.edge_weights2[(pair[0]-i, pair[1]-i)])
                
            indices_for_new_session.append(i)
            list1 = list(np.where(cell_types[i:i+len(list(mouse.graph.nodes))][mouse.graph.nodes]==0)[0])
            list2 = list(np.where(cell_types[i:i+len(list(mouse.graph.nodes))][mouse.graph.nodes]==1)[0])
            
            cell_types_ = loaded_dict[ind]["cell_types"]
            #if (list1 ==[] or list2 == []) and (list1 !=[] or list2 != []):
            if list1!=[]:
                ind_0.append(list(i + np.array(list1)))
            if list2 !=[]:
                ind_1.append(list(i + np.array(list2)))
            
            i += len(list(mouse.graph.nodes))
            print("mouse graph nodes", len(list(mouse.graph.nodes)))
            print("counter so far", i)
        
            print("number of cell types labeled", (cell_types[i-len(list(mouse.graph.nodes)):i][mouse.graph.nodes]!=-10).sum())
            print("number of cell types labeled 0: ", (cell_types[i-len(list(mouse.graph.nodes)):i][mouse.graph.nodes]==0).sum())
            print("number of cell types labeled 1: ", (cell_types[i-len(list(mouse.graph.nodes)):i][mouse.graph.nodes]==1).sum())
        
            print("ind_0", ind_0)
            print("ind_1", ind_1)
        
            print(indices_for_new_session)
        
        Graph_all = np.concatenate(Graph_all)
        Directed_Graph_all = np.concatenate(Directed_Graph_all)
        edge_weights = np.array(edge_weights)
        edge_weights2 = np.array(edge_weights2)
        edge_weights2 = np.squeeze(edge_weights2)
        edge_weights2 = edge_weights2[:,1:]

        Graph_all = torch.from_numpy(Graph_all).transpose(0,1).type(torch.LongTensor)
        Directed_Graph_all = torch.from_numpy(Directed_Graph_all).transpose(0,1).type(torch.LongTensor)
        edge_weights = torch.from_numpy(edge_weights)
        edge_weights2 = torch.from_numpy(edge_weights2)

        return Graph_all, Directed_Graph_all, edge_weights, edge_weights2, ind_0, ind_1, indices_for_new_session, mice
    

def load_filters_waveforms_isis():

            spike_filters = np.load("spike_filters.npy")
            running_filters_ = np.load("running_filters.npy")
            
            data = np.load('all_good_units.npz', allow_pickle=True)
            waveforms = data['all_waveforms']
            cell_types_ = data['cell_types']
            opsin = data['opsin']
            unique_types, all_type_counts = np.unique(cell_types_, return_counts=True)
            all_spike_times = data['all_spike_times']
            all_firing_rate = data['all_firing_rates']
            
            tagged_inds = (cell_types_!='untagged') & (cell_types_!='ChAt') # no chat because there's so few
            ind_tagged = np.where(tagged_inds)[0]
            tagged_waveforms = waveforms[tagged_inds]
            tagged_spike_times = all_spike_times[tagged_inds]
            tagged_firing_rate = all_firing_rate[tagged_inds]
            tagged_type = cell_types_[tagged_inds]
            
            cell_type_labels = {'D1' : 0,
                                'D2' : 1}

            running_filters = []
            filt_waveforms = []
            filt_isis = []
            filt_firing_rates = []
            filt_type = []
            ISI_features = []
    
            ind_0_diffData = []
            ind_1_diffData = []
            ind_non_select = []
            ind_select = []
            
            
            ISIs_max = -np.inf
            ISIs_min = np.inf
            for ind_cell in range(len(cell_types_)):
            
                this_spike_times = all_spike_times[ind_cell]
                ISIs = 1000*np.diff(this_spike_times)
            
                if len(ISIs)<=1:
                    continue
                if ISIs_max < ISIs.max():
                    ISIs_max = ISIs.max()
                if ISIs_min > ISIs.min():
                    ISIs_min = ISIs.min()
            time_bins = list(np.arange(0,100,1))
            time_bins.append(ISIs_max)
            time_bins = np.array(time_bins)
            running_bins = list(np.arange(np.percentile(running_filters_, 5), np.percentile(running_filters_, 95), (np.percentile(running_filters_, 95) -  np.percentile(running_filters_, 5))/100))
    
            for ind_cell in range(len(cell_types_)):
            
                this_waveform = waveforms[ind_cell]
                this_spike_times = all_spike_times[ind_cell]
                peak = np.min(this_waveform)
                peak_ind = np.where(this_waveform==peak)
                
                # unit not too close to tip and enough spikes to make good isi hist
                if (peak_ind[0][0] >= 10) and (len(this_spike_times) > 500) and (tagged_inds[ind_cell]):
                    ind_select.append(ind_cell)
                    if cell_types_[ind_cell] == "D1":
                        ind_0_diffData.append(ind_cell)
                    elif cell_types_[ind_cell] == "D2":
                        ind_1_diffData.append(ind_cell)
                elif tagged_inds[ind_cell]:
                    ind_non_select.append(ind_cell)
                    
                if peak_ind[0][0] < 10:
                    this_waveform = np.concatenate((np.zeros((10-peak_ind[0][0], this_waveform.shape[1])), this_waveform), axis=0)
                    peak_ind[0][0] = 10
                
                twod_waveform = this_waveform[peak_ind[0][0]-10:peak_ind[0][0]+10:2,peak_ind[1][0]-20:peak_ind[1][0]+60]
                
                twod_normed = twod_waveform/(-peak)
                filt_waveforms.append(twod_normed.flatten())
            
                ISIs = 1000*np.diff(this_spike_times)
                if len(ISIs) <=1:
                    h = np.zeros(100)
                else:
                    h,v = np.histogram(ISIs, time_bins, density=True)
                filt_isis.append(h)
                running_filters.append(np.histogram(running_filters_[ind_cell,:], running_bins, density = True)[0])

                filt_firing_rates.append(all_firing_rate[ind_cell])
                filt_type.append(cell_types_[ind_cell]) 
                ISI_features.append(ISIs)
            
            filt_firing_rates = np.array(filt_firing_rates).reshape((len(filt_firing_rates),1))
            clustering_data = np.concatenate((np.array(filt_waveforms),np.array(filt_isis),filt_firing_rates), axis=1)
            #clustering_data =  np.array(spike_trains_)
            filt_type = np.array(filt_type)
            filt_isis = np.array(filt_isis)
            running_filters = np.array(running_filters)
    
            return clustering_data, filt_waveforms, filt_isis, filt_firing_rates, spike_filters, running_filters, cell_types_, ISI_features


def load_graph_DataLoader(cell_types, full_data, seed):

        g = torch.Generator()
        g.manual_seed(seed)

        batch_size = 1
        all_nodes = np.arange(len(cell_types))
        input_nodes = torch.tensor(all_nodes[full_data.train_mask].tolist()).type(torch.LongTensor)
        
        train_loader = NeighborLoader(
            full_data,
            num_neighbors=[30] * 2,
            batch_size=batch_size,
            input_nodes=input_nodes)
            #num_workers=4,
            #worker_init_fn=seed_worker, 
            #generator=g)
        
        
        train_loader2 = NeighborLoader(
            full_data,
            num_neighbors=[30] * 2,
            batch_size=len(input_nodes),
            input_nodes=input_nodes)
            #num_workers=4,
            #worker_init_fn=seed_worker, 
            #generator=g
            

        input_nodes = torch.tensor(all_nodes[full_data.val_mask].tolist()).type(torch.LongTensor)

        valid_loader = NeighborLoader(
            full_data,
            num_neighbors=[30] * 2,
            batch_size=batch_size,
            input_nodes=input_nodes)
            #num_workers=4,
            #worker_init_fn=seed_worker, 
            #generator=g
            
         
        input_nodes = torch.tensor(all_nodes[full_data.test_mask].tolist()).type(torch.LongTensor)
        
        test_loader = NeighborLoader(
            full_data,
            num_neighbors=[30] * 2,
            batch_size=len(input_nodes),
            input_nodes=input_nodes)
            #num_workers=4,
            #worker_init_fn=seed_worker, 
            #generator=g
        
        return train_loader, train_loader2, valid_loader, test_loader, batch_size
        
                    
def data_loader(x, cell_types, train_mask, test_mask_labeled, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    
    batch_size = 1
    #data loader when we're not using graph
    data_train = torch.utils.data.TensorDataset(torch.tensor(x[train_mask]), torch.tensor(cell_types[train_mask]))
    data_test = torch.utils.data.TensorDataset(torch.tensor(x[test_mask_labeled]), torch.tensor(cell_types[test_mask_labeled]))

    
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle = True, 
        #num_workers=num_workers,worker_init_fn=seed_worker, generator=g
    )
    
    train_loader2 = torch.utils.data.DataLoader(
        data_train, batch_size=len(data_train), shuffle = True, 
        #num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
     
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=len(data_test), shuffle = True, 
        #num_workers=num_workers,worker_init_fn=seed_worker, generator=g
    )
    
    return train_loader, train_loader2, test_loader, batch_size

        