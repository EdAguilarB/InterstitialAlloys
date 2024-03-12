import numpy as np
from torch_geometric.utils import from_networkx, to_networkx
import itertools
import random
from copy import copy


from torch_geometric.loader import DataLoader

from icecream import ic


def get_coordinate(arr):
    
    if len(arr) != 3:
        raise ValueError("Input array must have a length of 3.")

    if abs(arr[0]) == 2 or abs(arr[0]) == 6:
        sign = '-' if arr[0] == -2 or arr[0] == 6 else '+'
        return f'{sign}x'
    elif abs(arr[1]) == 2 or abs(arr[1]) == 6:
        sign = '-' if arr[1] == -2 or arr[1] == 6 else '+'
        return f'{sign}y'
    elif abs(arr[2]) == 2 or abs(arr[2]) == 6:
        sign = '-' if arr[2] ==-2 or arr[2] == 6 else '+'
        return f'{sign}z'
    else:
        raise ValueError("Input array must contain at least one non-zero element.")

def count_directions(array):
    directions = []  # Create an empty list to store the directions
    frequencies = []  # Create an empty list to store the frequencies

    direction_counts = {}  # Create a dictionary to store the counts

    for entry in array:
        # Extract the direction (last character) from each entry
        direction = entry[-1]
        
        # Use the direction as the key in the dictionary and update the count
        if direction in direction_counts:
            direction_counts[direction] += 1
        else:
            direction_counts[direction] = 1

    # Convert the dictionary to NumPy arrays
    for direction, count in direction_counts.items():
        directions.append(direction)
        frequencies.append(count)

    return np.array(directions), np.array(frequencies)

def process_dictionary_values(data_dict):
    """
    generates boolean list of Mo and C atoms to remove given a dictionary
    """


    Mo = []  # List for keys with non-empty lists
    C = []   # List for flattened values

    for key, value in data_dict.items():
        if value:  # Check if the value is not an empty list
            Mo.append(key)

            if len(value) == 1 and isinstance(value[0], np.ndarray):
                # If there is a single numpy array, flatten and extend 'C'
                C.extend(value[0].flatten())
            elif all(isinstance(arr, np.ndarray) for arr in value):
                # If there are multiple numpy arrays, randomly choose one, flatten, and extend 'C'
                chosen_array = random.choice(value)
                C.extend(chosen_array.flatten())

    return Mo, C

def process_dictionary_values_no_overlap(data_dict):
    #random.seed(123456789)

    Mo = []  # List for keys with non-empty lists
    C = []   # List for flattened values
    atom_indexes_set = set()  # Set to track unique atom indexes in 'C'

    for key, value in data_dict.items():
        if value:  # Check if the value is not an empty list
            Mo_key = key  # Key to potentially add to 'Mo'
            np_arrays_to_add = []  # NumPy arrays to add to 'C'

            for np_array in value:
                # Check if any of the atom indexes in the array are already in 'C'
                if not any(idx in atom_indexes_set for idx in np_array):
                    np_arrays_to_add.append(np_array)

            if np_arrays_to_add:
                # Choose one random NumPy array from the list, if there are multiple
                chosen_np_array = random.choice(np_arrays_to_add)
                C.extend(chosen_np_array.flatten())
                Mo.append(Mo_key)
                atom_indexes_set.update(chosen_np_array)

    return Mo, C


def find_patterns(Mo_list, C_list, edge_dir, shape ='lin', overlap=True):

    """
    generates boolean list that tells whether an atom must be removed 
    or not based on a pattern that wants to be removed

    Args:
    Mo_list: list of Mo atoms that are involved in a Mo-C edge
    C_list: list of C atoms that are involved in a Mo-C edge
    edge_dir: list of directions of the Mo-C edges
    shape: pattern to be masked out

    Return:
    remove boolean list, which is True in the nth entry if the 
    nth Mo-C edge has to be removed or False if not
    """

    
    # list that will almacenate remove booleans
    remove = []

    # dictionary that contains in each key the Mo atom's index and in its values
    # a list of np arrays that contain the Cs index that 
    Mo_dict={}

    # list that contains the indexes of edges that correspond to a particular sub-fragment
    edge_idx_frag = []

    #counter of total subgraphs found within a structure
    total_subgraphs = 0

    # all Mo are analised
    # np.unique makes sure that if a Mo is involved in various edges,
    # it is studied only once
    for Mo in np.unique(Mo_list):
            
            # edges stores the directions of all the edges 
            # for which the Mo being analysed is involved
            edges = edge_dir[Mo_list==Mo]

            _, counts = np.unique(edges, return_counts=True)
            has_duplicates = any(counts > 1)
            if has_duplicates:
                  print("There is at least one edge with repeated direction for atom {}".format(Mo))
                  
            #Cs contains the carbon idx of those carbons sharing an edge with the central Mo
            Cs = C_list[Mo_list==Mo]

            #will store the direction of the edges involved in a given shape
            edge_idx_frag = []
            # will store carbons involved in a certain geometry for a central Mo 
            carbons = []
            
            # count_directions identifies the different direction of the Mo edges and count them
            # unique_dir contains the unique directions for all the edges that the Mo is involved 
            # and count contains the frequency of each direction
            unique_dir, count = count_directions(edges)

            # MoC shape is C-Mo system
            if shape == 'MoC': 

                rem_dir = unique_dir 
                
                for carbon, edge_direction in zip(Cs, edges):
                    edge_idx_frag.append(np.array(edge_direction))
                    carbons.append(np.array(carbon))

            # if statement checks what geometry wants to be masked out
            # lin shape is C-Mo-C system forming a 180 angle
            elif shape == 'lin':  
                # rem dir identifies if an unique element occurs less than 2 times, that is, 
                # there are two edges in the same direction (180 degrees angle)
                rem_dir = unique_dir[count==2]
                for lin in rem_dir:
                    mask = np.array([entry.endswith(lin) for entry in edges])
                    edge_idx_frag.append(np.array(edges[mask]))
                    carbons.append(np.array(Cs[mask]))

            # L shape is a C-Mo-C system forming a 90 angle (ortogonal)
            elif shape == 'L':
                # checks if there is more than one unique direction, meaning there is at least 
                # two edges with diferent direction, which means they are orthogonal, meaning 
                # that always that there are more than 2 directions, the statement below is true
                if len(unique_dir) >1:
                        # if there is at least one pair of orthogonal edges, then they have to be removed. If there are more than two edges,
                        # does not matter whether there are two repeated directions or if they all have different direction, all of them must 
                        # be removed since each of them is orthogonal to at least one other edge
                        rem_dir = unique_dir
                        for l1 in range(len(edges)-1):
                            for l2 in range(l1,len(edges)):
                                if edges[l1][1] != edges[l2][1]:
                                    edge_idx_frag.append(np.array((edges[l1],edges[l2])))
                                    carbons.append(np.array([Cs[l1], Cs[l2]]))
                else:
                      rem_dir = None

        
            # fT stands for flat T shape, which is a Mo with 3 carbons forming a T
            elif shape == 'fT':
                # checks if there is more than one unique direction, meaning there is at least 
                # two orthogonal edges. Also checks if there is an unique element 
                # which count is two, which means that there is 180 angle C-Mo-C system
                # if there is a pair of edges that are ortogonal and two that are parallel, 
                # then there must be a flat T shape in the central Mo

                # Note: if there are only 3 elements, the if statement bellow will do fine in 
                # analysing the relative position of the carbons, but if there are more than 3
                # edges, then the condition is always true because of geometry
                if len(unique_dir) >= 2 and len(count[count==2]) >= 1:
                    rem_dir = unique_dir
                    for lin in unique_dir[count==2]:
                        lin_mask = np.array([entry.endswith(lin) for entry in edges])
                        for l in edges[~lin_mask]:
                            dirs = np.concatenate([edges[lin_mask], np.expand_dims(l,axis=0)])
                            edge_idx_frag.append(dirs)
                            carbons.append(Cs[np.isin(edges, dirs)]) 

                else:
                      rem_dir = None


            # pyr stands for pyramid, which is a shape where a central Mo share an edge with 
            # 3 different carbons where each pair of carbons have an angle of 90 for all the C-Mo-C 
            # system. for that to happen, the inly condition is that the central Mo has 3 edges with 
            # 3 different directions, meaning unique_dir must be 3 for that to happen
            elif shape == 'pyr':
                if len(unique_dir) == 3:
                        rem_dir = unique_dir
                        pyr = [
                                np.array((edges[i], edges[j], edges[k]))
                                for i in range(len(edges) - 2)
                                for j in range(i + 1, len(edges) - 1)
                                for k in range(j + 1, len(edges))
                                if edges[i][1] != edges[j][1] and edges[j][1] != edges[k][1] and edges[i][1] != edges[k][1]
                                    ]
                        edge_idx_frag.extend(pyr)

                        for pyr_shape in pyr: 
                            carbons.append(Cs[np.isin(edges, pyr_shape)])
                else:
                      rem_dir = None

            #cross shape is a central Mo with 4 C around it in the same plane, forming a cross. 
            # To check if that fragment does exist in a central molybdenum, condition bellow 
            # checks if there is at least 2 different edge directions and if there are at least
            # two directions that are repeated twice, meaning that the system forms a cross    
            elif shape == 'cross':
                if len(count[count==2]) >= 2:
                        rem_dir = unique_dir[count==2]
                        for lin1 in range(len(rem_dir)-1):
                            lin1_mask = np.array([entry.endswith(rem_dir[lin1]) for entry in edges])
                            for lin2 in range(lin1+1, len(rem_dir)):
                                lin2_mask = np.array([entry.endswith(rem_dir[lin2]) for entry in edges])
                                dirs = np.concatenate([edges[lin1_mask], edges[lin2_mask]])
                                edge_idx_frag.append(dirs)
                                carbons.append(Cs[np.isin(edges, dirs)])
                else:
                      rem_dir = None

            # tetra is a shape where a central Mo is sourounded by 4 carbons which are not in the same plane
            # to check if that geometry does exist, there must exist 3 different edge directions, and at least
            # one direction is repeated
            elif shape == 'tetra':
                if len(unique_dir) == 3 and len(count[count==2]) >= 1:
                        rem_dir = unique_dir
                        combinations = list(itertools.combinations(edges, 4))
                        for combo in combinations:
                            direction, times = count_directions(combo)
                            if len(direction[times == 2]) == 1:
                                combo = np.array(list(combo))
                                edge_idx_frag.append(combo)
                                carbons.append(Cs[np.isin(edges, combo)])
                else:
                      rem_dir = None

            # a central Mo connected to 5 carbons
            elif shape == 'penta':
                if len(edges) >= 5:
                        rem_dir = unique_dir
                        combinations = list(itertools.combinations(edges, 5))
                        for comb in combinations:
                            comb = np.array(list(comb))
                            edge_idx_frag.append(comb)
                            carbons.append(Cs[np.isin(edges, comb)])
                else:
                      rem_dir = None
                

            # a central Mo connected to 6 C
            elif shape == 'hexa':
                if len(count[count==2]) == 3:
                        rem_dir = unique_dir
                        edge_idx_frag.append(edges)
                        carbons.append(Cs)
                else:
                      rem_dir = None


            else:
                 print('No valid figure was provided. No permutation will be made to the graph.')



            # np.isin checks if the elements of the list edges are in the rem_dir array
            # if the nth entry in edges is in rem_dir, then the nth entry will return True

            Mo_dict[Mo] = carbons

            total_subgraphs += len(carbons)


    if overlap:
        remove = process_dictionary_values(data_dict=Mo_dict)
    
    else:
        remove = process_dictionary_values_no_overlap(data_dict=Mo_dict)

    total_masked_substructures = len(remove[0])

    print(f'A total of {total_subgraphs} matching subpatterns were found for {shape} shape.')

    
    return remove, total_masked_substructures

def remove_pattern(graph, shape = 'lin', remove = 'all', overlap=True, compare_shape = None):
    
    """
    finds a given pattern within a molybdenum carbide super cell and removes it

    Args:
    graph (torch_geometric graph): graph to be modified
    shape: shape to be found within cell and that will be removed

    Returns:
    torch_geometric graph without the pattern that was sought

    """

    # information that is going to be lost in the convertion of the graph is 
    # stored in variables so that it can be recovered in the future
    y = graph.y
    file_name = graph.file_name

    #converts the graph to a networkx instance, itom identity is passed as node features
    g = to_networkx(graph, node_attrs=['embeddings'], edge_attrs=['edge_attr'])

    #adds information of the coordinates of each atom in the cell and it's identity 
    for node in g.nodes:
        g.nodes[node]['coords'] = graph.coords[node].astype(int)
        g.nodes[node]['x'] = graph.x[node]


    #lists to almacenate the number of the Mo and C involved in an edge
    #the nth Mo shares and edge with the nth C in the Mo_list and C_list
    #edge dir will store the nth direction of the edge between the nth Mo and nth C
    Mo_list = []
    C_list = []
    edge_dir = []

    #iterates over all edges within the graph
    for edge in g.edges():

        #checks if the first atom is Mo and the second C
        #if that is the case, then the algorithm found a Mo-C edge
        if g.nodes[edge[0]]['x'][0] == 1 and g.nodes[edge[1]]['x'][0] == 0:
            #the Mo, C are appended to the list. direction of edge is appended as well
            Mo_list.append(edge[0])
            C_list.append(edge[1])
            edge_dir.append(get_coordinate(g.nodes[edge[0]]['coords'] - g.nodes[edge[1]]['coords']))
    
    
    #transforms lists in np.arrays for further analysis
    Mo_list, C_list, edge_dir = np.array(Mo_list), np.array(C_list), np.array(edge_dir)

    #find_patterns function finds the patterns that must be masked. Returns a list of booleans
    #that tell whether a node has to be deleted (True) or not (False)
    mask, len_fragments = find_patterns(Mo_list=Mo_list, C_list=C_list, edge_dir=edge_dir, shape=shape, overlap=overlap)

    #Mo_remove and C_remove contain the atoms that have to be eliminated
    Mo_remove = mask[0]
    C_remove = mask[1]

    if compare_shape:

        print('Comparing shape being used: {}'.format(compare_shape))
        mask_i, _ = find_patterns(Mo_list=Mo_list, C_list=C_list, edge_dir=edge_dir, shape=compare_shape)

        Mo_remove = list(set(Mo_remove).intersection(mask_i[0]))

        len_fragments = len(Mo_remove)

        C_remove = mask_i[1]

    print(f'Total of fragments being removed: {len_fragments}')

    #we create a new graph ng that will be the one permutated
    #g is not permutated since change in size raises an error when looping
    ng = g.copy()

    #iterates over all nodes of g and in case the node is in the list Mo_remove 
    #or C_remove, it gets eliminated from ng
    print('Masking {} atoms'.format(remove))
    for node in g.nodes:
        if (node in Mo_remove) and (remove == 'all' or remove == 'Mo'):
            ng.remove_node(node)
        if (node in C_remove) and (remove == 'all' or remove == 'C'):
            ng.remove_node(node)

    #the permutated networkx graph ng is turned back into a torch_geometric graph
    torch_graph = from_networkx(ng)

    #lost information in transformation is recovered
    torch_graph.y = y
    torch_graph.file_name = file_name
    
    return torch_graph, len_fragments


def permute_graphs(batch, shape = 'lin', remove = 'all', overlap=False, compare_shape = None):

    """
    permutes a batch of graphs to convert them into new masked graphs

    Args:
    batch (torch_geometric DataLoader batch): batch to be masked
    shape: shape to be found within cell and that will be removed
    remove (str): type of masking to apply. If set to 'all', the entire fragment will be removed,
    if set to 'Mo', only the Molybdenum within the fragment will be removed, and if set to 'C',
    only the carbons will be removed

    Returns:
    batch with permutated graphs
    """

    # list that will contain the permutated graphs
    new_dataset = []

    # embeddings are stored in a variable so it does not get lost when 
    # converting from batch to graph data

    embeddings = batch.embeddings

    #iterates over the graphs of the batch
    for i in range(batch.num_graphs):

        # copy the ith graph of the batch in the graph variable
        graph = copy(batch[i])

        # recover the embeddings information from the batch 
        # and put it into the graph
        graph.embeddings = embeddings

        # graph is put into remove_pattern function which will 
        # take out all the patterns within the cell
        new_graph, length = remove_pattern(graph, shape, remove, overlap, compare_shape)

        #the permuted graph is appended to the new_dataset list
        new_dataset.append(new_graph)

    
    # create a new loader with the permuted graphs 
    new_loader = DataLoader(new_dataset, batch_size = len(new_dataset), shuffle = False)

    #returns the batch with permutated graphs
    for batch in new_loader:
        return batch, length