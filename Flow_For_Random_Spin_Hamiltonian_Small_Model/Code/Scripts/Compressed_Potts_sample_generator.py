import numpy as np 
import copy

###### 

## This script consists of two classes 
    #1. The 'Parameter_dictionary' class
    #2. The ' Compressed_Potts_Sample_Generator' class

## ---------------- Parameter_dictionary class ------------
## First we construct a parameter dictionary saver
## this will save various parameters in json file
## The things to save are:
#### - Partition map
#### - Compressed Vocabulary at every position
#### - Model Weights 
#### - Model Bias
## This class will combine all this information into a dictionary object to be saved as json



#### ---------------- Compressed Potts class --------------
## Takes in the parameter dictionary (json file)
## This json file will consist of 
## We will have to open this json file to create the dictionary that the sample generator will use
## Hence we create this json dictionary reader
#### - Partition map
#### - Compressed Vocabulary at every position
#### - Model Weights 
#### - Model Bias

### Opens these and assigns them to internal variables.
### Generates samples using MCMC inside each partition with influence from adjacent lower entropy partition
 



class Paramteter_dictionary_old():

    def __init__(self, Raw_Partition_Map, Paramteter_Dictionary =None):
        self.Partition_Map = Raw_Partition_Map
        self.num_partition, self.num_features = np.shape(Raw_Partition_Map)

        if type(Paramteter_Dictionary) == type(None):
            ## construct a dictionary
            self.Parameter_Dictionary={}
            self.Parameter_Dictionary["Partition_Map"] = Raw_Partition_Map
        

    def Save_Vocabulary(self, Compressed_dictionary):

        self.Compressed_Vocab ={}
        self.Compressed_Vocab["Vocabulary_size"] = Compressed_dictionary[0]

        self.Compressed_Vocab["Words_observed"] = Compressed_dictionary[1]

        Full_vocab = np.unique(np.concatenate(list(Compressed_dictionary[1].values())))

        self.Compressed_Vocab["Full_Vocabulary"] = Full_vocab[0:-1]
        self.Compressed_Vocab["Extra_word"] = Full_vocab[-1]

        #for k in self.Compressed_Vocab["Words_observed"].keys():
        #    Words_unobserved = (np.setdiff1d(Unique,Model.Compressed_dictionary()[1][0]))
        #self.Compressed_Vocab["Words_missing"]

        self.Parameter_Dictionary["Vocabulary_Details"] = self.Compressed_Vocab
        
        return self.Compressed_Vocab
    
    def Save_Weights(self, Weights_dictionary):
        ### check if we have same number of weight matrix:
        m1 = len(Weights_dictionary.keys())
        if m1 != self.num_partition:
            print("number of partition not equal to number of weight matrices.")
        else:
            self.Weights = Weights_dictionary
            self.Parameter_Dictionary["Weights"] = Weights_dictionary
        
    def Save_bias (self, Bias_dictionary):
        ### check if we have same number of bias vectors:
        m1 = len(Bias_dictionary.keys())
        if m1 != self.num_partition:
            print("number of partition not equal to number of bias vectors.")
        else:
            self.Bias = Bias_dictionary
            self.Parameter_Dictionary["Bias"] = Bias_dictionary
    
    def Return_Dict(self):
        return self.Parameter_Dictionary
    
    
class Paramteter_dictionary():


    def __init__(self, Model):
        self.Model = Model

    
        self.Raw_Partition_Map = self.Model.Raw_Partition_Map
        self.num_partition, self.num_features = np.shape(self.Raw_Partition_Map)

        ## construct a dictionary
        self.Parameter_Dictionary={}
        self.Parameter_Dictionary["Partition_Map"] = self.Raw_Partition_Map


        self.Detailed_Partition_Map = self.Model.Partition_Map

        self.Save_Vocabulary(self.Model.Compressed_dictionary())

        self.Save_Weights(self.Model.Weights)
        
        self.Save_bias(self.Model.Bias)


        

    def Save_Vocabulary(self, Compressed_dictionary):

        self.Compressed_Vocab ={}
        self.Compressed_Vocab["Vocabulary_size"] = Compressed_dictionary[0]

        self.Compressed_Vocab["Words_observed"] = Compressed_dictionary[1]

        Full_vocab = np.unique(np.concatenate(list(Compressed_dictionary[1].values())))

        self.Compressed_Vocab["Full_Vocabulary"] = Full_vocab[0:-1]
        self.Compressed_Vocab["Extra_word"] = Full_vocab[-1]

        #for k in self.Compressed_Vocab["Words_observed"].keys():
        #    Words_unobserved = (np.setdiff1d(Unique,Model.Compressed_dictionary()[1][0]))
        #self.Compressed_Vocab["Words_missing"]

        self.Parameter_Dictionary["Vocabulary_Details"] = self.Compressed_Vocab
        
        return self.Compressed_Vocab
    
    def Save_Weights(self, Weights_dictionary):
        ### check if we have same number of weight matrix:
        m1 = len(Weights_dictionary.keys())
        if m1 != self.num_partition:
            print("number of partition not equal to number of weight matrices.")
        else:
            self.Weights = Weights_dictionary
            self.Parameter_Dictionary["Weights"] = Weights_dictionary
        
    def Save_bias (self, Bias_dictionary):
        ### check if we have same number of bias vectors:
        m1 = len(Bias_dictionary.keys())
        if m1 != self.num_partition:
            print("number of partition not equal to number of bias vectors.")
        else:
            self.Bias = Bias_dictionary
            self.Parameter_Dictionary["Bias"] = Bias_dictionary
    
    def Return_Dict(self):
        return self.Parameter_Dictionary

    
#### ---------Compressed Potts Sample Generator --------

class Sample_Generator_Compressed_Potts():
    def __init__(self, Parameter_dictionary, Influence = True, Temperature = 1, noise = 1e-8):

        self.Parameter_dict = self.Json_dict_to_numpy(Parameter_dictionary, convert_keys_to_int=True)

        self.Raw_Partition_Map = self.Parameter_dict['Partition_Map']
        self.Temperature = Temperature
        self.noise = noise
        self.Influence = Influence

        
        Vocab_details = self.Parameter_dict['Vocabulary_Details']

        self.compressed_alphabet_size  = Vocab_details['Vocabulary_size']
        self.num_of_positions = len(self.compressed_alphabet_size)


        self.conversion_dictionary = Vocab_details['Words_observed']
        self.alphabet_list = Vocab_details['Full_Vocabulary']
        self.extra_character = Vocab_details['Extra_word']


        ###
        self.Weights = self.Parameter_dict["Weights"]
        self.Bias = self.Parameter_dict["Bias"]

        ##
        ###-- constructing the list of indices for each position ---
        self.Position_index = self.Position_index_finder()
        self.Partition_Map_Setter(self.Raw_Partition_Map)

        self.Partition_index_slice =self.Partition_Index_slicer()

        ###-
        self.fundamental_connectivity = self.Fundamental_Connectivity()

        self.unseen_words ={}
        for k in self.conversion_dictionary.keys():
            unseen = np.setdiff1d(self.alphabet_list, self.conversion_dictionary[k])
            self.unseen_words[k] = unseen


    def Partition_Map_Setter(self, Raw_Partition_Map):
        print("..Setting Partition Map..")
        print("Number of partitions =", len(Raw_Partition_Map))

        self.Raw_Partition_Map = Raw_Partition_Map

        Connected_indices ={}
        for i,part_i in enumerate(Raw_Partition_Map):
            Arg_p = np.argwhere(part_i==1).reshape(-1)
            conn_p =[]
            for a in Arg_p:
                conn_p.append(self.Position_index[a])

            conn_p = np.concatenate(conn_p)
            Connected_indices[i] = conn_p

        self.Partition_Map = Connected_indices

        
    def One_hot_encoder(self, Compact_state):
        """For any given state, it returns a one hot encoded state
        - unseen words at any position are mapped to unknown"""
        Dictionary = self.conversion_dictionary
        Dim = self.compressed_alphabet_size
        Full_Vector = []
        for i,word_i in enumerate(Compact_state):
            dim_i = Dim[i]
            Dict_i = Dictionary[i]
            Sub_vector = np.zeros(dim_i)

            if word_i in Dict_i:
                arg_one = (np.argwhere(Dict_i==word_i).reshape(-1).reshape(-1))
            else:
                ## word out of vocabulary. Map it to unknown
                arg_one = (np.argwhere(Dict_i==self.extra_character).reshape(-1))

            Sub_vector[arg_one] =1
            Full_Vector.append(Sub_vector)
        Full_Vector = np.concatenate(Full_Vector)
        return Full_Vector


    def One_hot_decoder(self, Unfurled_state):
        """for any one hot encoded state, it returns the compact state
         (list of words that make the sentence)"""
        Dictionary = self.conversion_dictionary
        Dim = self.compressed_alphabet_size
        Compact_state = []
        start_index = 0
        end_index = 0
        for i, dim_i in enumerate(Dim):
            end_index += dim_i
            Dict_i = Dictionary[i]
            sub_vector = Unfurled_state[start_index:end_index]
            arg_one= np.argwhere(sub_vector==1).reshape(-1)
            word_i = Dict_i[arg_one]

            if word_i == self.extra_character:
                word_i = np.random.choice(self.unseen_words[i], 1)
                
            start_index += dim_i
            Compact_state.append(word_i)

        return np.array(Compact_state).reshape(-1)
    

    def Position_index_finder(self):
        Position_index = {}
        start_p = 0
        for p in range(0, len(self.compressed_alphabet_size)):
            end_p = start_p + self.compressed_alphabet_size[p]
            Position_index[p] = np.arange(start_p, end_p)
            start_p = end_p
        return Position_index
    

    def Fundamental_Connectivity(self):

        Partition_Map_dict = self.Partition_Map
        Position_index = self.Position_index
        Raw_partition = self.Raw_Partition_Map
        Influence = self.Influence

        ### this doesn't have zeroing out of the diagonals

        Fundamental_connectivity_dict = {}
        partitions = list(Partition_Map_dict.keys())

        for p in partitions:
            #raw_part = Raw_partition[p]
            #positions_p = np.argwhere(raw_part==1).reshape(-1)
            Total_dof_p_row = len(Partition_Map_dict[p])
            Total_dof_p_col = Total_dof_p_row
            Total_dof_prev = 0

            if p > 0 and Influence:
                Total_dof_prev = len(Partition_Map_dict[p-1])
                Total_dof_p_col+=Total_dof_prev

            FC_p = np.ones((Total_dof_p_row, Total_dof_p_col))

            Fundamental_connectivity_dict[p] = FC_p

        ### Now we zero out the appropriate blocks that correspond to self interaction.
        for idx in partitions:
            Raw_part = Raw_partition
            full_part = Partition_Map_dict
            Position_index = Position_index
            Super_offset = 0
            if idx>0 and Influence:
                Super_offset = len((full_part[idx-1]).reshape(-1))
            arg_part = np.argwhere(Raw_part[idx]==1).reshape(-1)
            len_part = len(full_part[idx])
            offset = 0
            FC_i = Fundamental_connectivity_dict[idx]
            for p in arg_part:
                #print(p)
                args_p = np.arange(0,len(Position_index[p])) + offset 
                args_p_col = args_p + Super_offset

                #print(len(args_p))
                #print(np.shape(FC))
                v1 = np.zeros(len_part) 
                v1[args_p] = 1

                v2 = np.zeros(Super_offset+len_part)
                v2[args_p_col] = 1
                
                Matrix = np.outer(v1, v2)
                offset += len(Position_index[p])
                FC_i-= Matrix
        return Fundamental_connectivity_dict
    
    
    def Json_dict_to_numpy(self, data, convert_keys_to_int=False):
        """To load json file of parameters
            - json files cannot have numpy ndarrays
            - Hence value arrays are saved as lists
            - This converts it back to dictionary with ndarrays
            - Optionally converts string keys back to integers"""
        Data = copy.deepcopy(data)
        
        if type(Data) == dict:
            keys = list(Data.keys())  # Convert to list since we might modify dict
            new_dict = {}
            
            for key in keys:
                val = Data[key]
                
                # Convert key if needed
                if convert_keys_to_int:
                    try:
                        new_key = int(key)
                    except (ValueError, TypeError):
                        new_key = key  # Keep as string if conversion fails
                else:
                    new_key = key
                
                # Recursively process the value
                new_dict[new_key] = self.Json_dict_to_numpy(val, convert_keys_to_int)
            
            return new_dict
            
        elif type(Data) == list:
            return np.array(Data)
        else:
            return Data
        
    def Partition_Index_slicer(self):

        Raw_Partition = self.Raw_Partition_Map
        Compressed_alphabet_size = self.compressed_alphabet_size

        Index_slices={}
        for p in range(0,len(Raw_Partition)):
            partition = Raw_Partition[p]
            positions_p = np.argwhere(partition==1).reshape(-1)
            Alphabets_size_p = Compressed_alphabet_size[positions_p]
            cumulative_size = np.cumsum(Alphabets_size_p)
            start=0
            Slice_p={}
            for i in range(0, len(Alphabets_size_p)):
                end = int(cumulative_size[i])
                Slice_p[int(positions_p[i])] = [start, end]
                #Slice_p.append([start, end])
                start = end

            Index_slices[p] = Slice_p

        return Index_slices
        
    ###########
    def Random_state_generator(self, verbose = False):
        words = np.random.choice(self.alphabet_list, self.num_of_positions)

        one_hot_random = self.One_hot_encoder(words)
        if verbose:
            return one_hot_random, words
        else:
            return one_hot_random
    
    ####--- Data Partition ----

    def Data_part_by_index(self, partition_index):
        """Partitions the Unique Data on the given partition
        Input:
        partition_index(int): from 0 to len(Partition)
        """
        partition_index = int(partition_index)
        Data = self.Unique_datapoints
        Part_args = self.Partition_Map[partition_index]
        Data_part = Data[:,Part_args]
        return Data_part
    
    def Data_part(self):
        Partitions = list(self.Partition_Map.keys())
        Partitioned_Data = {}

        for p in Partitions:
            Partitioned_Data[p] = self.Data_part_by_index(p)
        return Partitioned_Data
    
    def Data_Partition(self, Data, partition_index):
        #Data = self.Unique_datapoints
        padded = False
        if len(np.shape(Data)) ==1:
            ## we will need to pad the other dimension for slicing
            Data = Data.reshape(1,-1)
            padded = True

        Row_args = self.Partition_Map[partition_index]

        if self.Influence and partition_index>0:
            Col_args = np.concatenate([self.Partition_Map[partition_index-1],self.Partition_Map[partition_index]])

        else:
            Col_args = self.Partition_Map[partition_index]

        Data_self = Data[:, Row_args]
        Data_interaction = Data[:, Col_args]

        if padded==True:
            Data_self = Data_self.reshape(-1)
            Data_interaction = Data_interaction.reshape(-1)

        return Data_self, Data_interaction
    
    ########
        ### ------------------------ Energy Calculators ---------------------

    ### ----------- Energy Calculators -----------
    def Energy_Array_given_partition(self, Weight_part, Bias_part, Data_self, Data_interaction):
        """Calculates energy for an Array using partitioned weight/bias matrices"""

        Energy_all = np.sum(Data_self @ Weight_part * Data_interaction , axis = 1) + Data_self@Bias_part
        return Energy_all


    def Energy_state_given_partition(self, Unfurled_Sequence, partition_index):
        
        """Calculates energy for a sequence using partitioned weight/bias matrices"""

        Data_self, Data_interaction = self.Data_Partition(Unfurled_Sequence, partition_index)

        Weight = self.Weights[partition_index]
        Bias = self.Bias[partition_index]
        
        Energy_Sequence = np.dot(Data_self @ Weight, Data_interaction) + np.dot(Bias, Data_self)
        
        return Energy_Sequence
    
    ####
        ### -----------------For activation Probabilities --------------------

    def Softmax_probability_stable(self, Energy_list, sign = 1, temperature=1):

        """----- stable softmax function that doesn't overflow ----
        --- if sign = 1 --> softmax of Energy_list (default)
        --- if sign = -1 --> softmin of Energy_list

        --- temperature scales energy by division (default temperature = 1)
        
        """

        Energy_array = np.array(Energy_list) * sign

        max_value = np.max(Energy_array)

        Energy_array_scaled = (Energy_array - max_value)/temperature

        likelihood_energy = np.exp(Energy_array_scaled)

        normalization = np.sum(likelihood_energy)

        Softmax = likelihood_energy/normalization

        return Softmax


    ####
    def Single_Mutants_Generator_Full(self, Unfurled_Sequence, partition_index):

        ## split the sequence into self and interaction components for this partition:
        ## include a copy of the original data

        Data_self, Data_interaction = self.Data_Partition(Unfurled_Sequence, partition_index)
        Diff = len(Data_interaction) - len(Data_self)
        Connectivity_partition = self.fundamental_connectivity[partition_index]
        Part_dof = len(Data_self)

        ### purely prev partition data:
        Data_prev_part = Data_interaction[:Diff]
        Connectivity_relevant = Connectivity_partition[:, Diff:]
        Single_mutant_full_self =  ((Connectivity_relevant * Data_self) + np.eye(Part_dof))
        num_mutants = len(Single_mutant_full_self)

        ### 
        D_prev = np.array([Data_prev_part for i in range(0, num_mutants)])
        Single_mutant_interaction = np.hstack([D_prev, Single_mutant_full_self])

        return Single_mutant_full_self, Single_mutant_interaction
    


    def Activation_Probability(self, Unfurled_Sequence, partition_index, temperature):
        ### I don't need to optimize this too much since I will be using this for sample generation only
        
        # get relevant portion of weight and bias
        Weight_part = self.Weights[partition_index]
        Bias_part = self.Bias[partition_index]

        ## making single mutations
        Data_self, Data_interact = self.Single_Mutants_Generator_Full(Unfurled_Sequence, partition_index)

        ## Now we get Energy of these mutants:
        E_mutants = self.Energy_Array_given_partition(Weight_part, Bias_part, Data_self, Data_interact)
        Probability_array = []  #<= Probability within the self seq. 

        ## for each position
        positions = np.argwhere(self.Raw_Partition_Map[partition_index] ==1).reshape(-2)
        start = 0
        Positions_end = np.cumsum(self.compressed_alphabet_size[positions])
        for i,p in enumerate(Positions_end):
            end = p
            #print(start, end)
            E_site = E_mutants[start:end]
            #print(np.sum(E_site))
            Prob_site = self.Softmax_probability_stable(E_site, sign=-1, temperature = temperature)
            #print(Prob_site)
            start = end
            Probability_array.append(Prob_site)
        Probability_array = np.concatenate(Probability_array)
        return Probability_array, E_mutants
    

    def Forward_Gibbs_Pass(self, Unfurled_Sequence, partition_index, Step_Split = 0.5, temperature=1 ):

        
        Positions_in_partition = list(self.Partition_index_slice[partition_index].keys())
        Index_slice_dict = self.Partition_index_slice[partition_index]

        Activation_prob, _ = self.Activation_Probability(Unfurled_Sequence, partition_index, temperature=temperature)

        Length = len(Positions_in_partition)
        
        Num_sites_to_change = int(Length*Step_Split)
        
        ## at least one site needs to change
        Num_sites_to_change = int(np.max((1, Num_sites_to_change)))

        ### get the indices of the sites to change

        Positions_to_change = np.sort(np.random.choice(Positions_in_partition, Num_sites_to_change, replace=False))
        #print(Positions_in_partition)
        #print("-->",Positions_to_change)

        #Seq_self, Seq_interact = self.Data_Partition(Unfurled_Sequence,partition_index)

        ### Initialize the Next State as as copy of the current state
        ### we will keep the indices that are not changing (by not operating on them)
        
        for pos in Positions_to_change:
            indices_pos = self.Position_index[pos]
        
            start, stop = Index_slice_dict[pos]
            num_alphabets = stop-start

            Prob_i = Activation_prob[start:stop]
            ## pick the new state in this position that pings( that is 1)
            Picked_state = np.random.choice(range(0,num_alphabets), p=Prob_i)
            ## blank out the previous state at this site
            Next_state = np.zeros(len(indices_pos))
            ## replace it with the new state
            Next_state [Picked_state]=1
            
            Unfurled_Sequence[indices_pos] = Next_state

        return Unfurled_Sequence
    

    
    def Gibbs_Sampling_Partitioned(self, State, partition_index, Num_iterations, Step_Split = 0.5, temperature = 1):
        State = State.copy()
        for i in range(0, Num_iterations):
            State = self.Forward_Gibbs_Pass(State,partition_index,  Step_Split, temperature=temperature)
        return State
    

    def Gibbs_Sampling_through_Layers(self, Num_iterations, Step_Split=0.5, temperature = 1):

        Random_State = self.Random_state_generator()

        Final_State = Random_State.copy()
        for partition_index in range(0,len(self.Raw_Partition_Map)):
            #Row_args, Col_args = self.Relevant_Partition_Args(p_index)
            #W_part, B_part = self.Weight_Bias_Partition(Row_args, Col_args)
            Final_State = self.Gibbs_Sampling_Partitioned(Final_State, partition_index, Num_iterations, Step_Split, temperature=temperature)
        return Final_State, Random_State
    

    #### Let us also include Parallel Tempering:

    def PT_Temperature_Steps(self, num_chains , min_temp = 0.5, max_temp = 5, type = 1):

        """
        - Given
        num_chains = number of chains to create during parallel tempering
        min_temp = smallest temperature for the chain
        max_temp = largest temperature for the chain
        
        type: 1,2,3,4
        1 --> Inverse-linear steps in temperature between min_temp and max_temp
                (linear steps in inverse temperature)
                - This performs the best in my experience
                
        2--> Geometric steps between min_temp and max_temp
                - This is also good

        3--> Linear steps between min_temp and max_temp
                - this is pretty poor. Mostly high energy phases explored

        4--> Same temperaure for all chains.
                - All chains set at T=1
                - This is to explore the space using walkers of same energy (step size)
        """

        ## sort and regularize the temperatures
        min_temp, max_temp = np.sort([min_temp, max_temp])  +1e-6
        


        if type==1:
            ### Inverse-linear temperature scheme
            ### linear steps in beta (the inverse temperature)

            beta_low_temp = 1/min_temp
            beta_high_temp = 1/max_temp

            beta_steps = np.linspace(beta_low_temp, beta_high_temp, num_chains)

            Temp_steps = 1/beta_steps


        if type ==2:
            ### Geometric temperature steps scheme

            ratio = (max_temp/min_temp)**(1/(num_chains-1))
            r_step = np.arange(0, num_chains)

            Temp_steps = min_temp *(ratio **r_step)


        if type ==3:
            ### linear temperature steps scheme
            Temp_steps =  np.linspace(min_temp, max_temp, num_chains)


        if type ==4:
            #all temperatures same
            ## here all we are doing is exploring the space with walkers of same energy
            Temp_steps = np.ones(shape = num_chains)

        return Temp_steps
    


    def Gibbs_Sampling_through_Layers_with_Parallel_Tempering(self, State, partition_index, Num_iterations, temperature_list, Swap_iteration = 10, Cool_iterations = 30,Step_Split = 0.5,):

        num_chains = len(temperature_list)
        State = State.copy()

        ##
        Weight = self.Weights[partition_index]
        Bias = self.Bias[partition_index]

        # Initialize chains
        chains = [State.copy() for _ in range(num_chains)]

        # Initialize energies
        energies = np.zeros((num_chains, len(State)))

        # Run parallel tempering
        for step in range(Num_iterations):
            ## update all chains-->forward gibbs
            for i in range(num_chains):
                # Perform Gibbs sampling
                temp_for_chain = float(temperature_list[i])
                chains[i] = self.Forward_Gibbs_Pass(chains[i], partition_index=partition_index, Step_Split=Step_Split, temperature=temp_for_chain)
            
            # Swap states between chains with different temperatures
            # swap every "Swap-iteration"
            if np.mod(step,Swap_iteration)==0 and step >1:
                for i in range(num_chains - 1):
                    ## Compute energy for chain:
                    Chain_self, Chain_interact = self.Data_Partition(np.array(chains), partition_index)
                    energies = self.Energy_Array_given_partition(Weight_part=Weight, Bias_part=Bias, Data_self=Chain_self, Data_interaction=Chain_interact)
                    
                    ## choose different chain to exchange with
                    ## here next temperature step chain is chosen
                    j = i+1

                    # Calculate swap probability
                    # Avoid the overflow issue by hardcoding the power in swap exponential (exp of difference between things converges better)
                    Power = (energies[i] - energies[j]) / (temperature_list[i] - temperature_list[j])
                    if Power > 0:
                        swap_factor = 1
                    else:
                        swap_factor = np.exp(Power)

                    if np.random.rand() < swap_factor:
                        chains[i], chains[j] = chains[j], chains[i]

        ## Run cool down for "cool iteration":
        ## here we explore at the model temperature.
        ## this is the final finetuning
        for i in range(num_chains):
            temp_for_chain = self.Temperature
            chains[i] = self.Gibbs_Sampling_Partitioned(chains[i], partition_index, Cool_iterations, Step_Split, temp_for_chain)
        ## the states have been swapped around.
        ## hence we will compute the energy one last time
        Chain_self, Chain_interact = self.Data_Partition(np.array(chains), partition_index)
        energies = self.Energy_Array_given_partition(Weight_part=Weight, Bias_part=Bias, Data_self=Chain_self, Data_interaction=Chain_interact)
            
        return chains, energies



    
