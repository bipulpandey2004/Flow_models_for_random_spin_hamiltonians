import numpy as np
from tqdm import tqdm


#######
class Compressed_Potts_Quadratic:
    def __init__(self, Compact_Training_Data, Partition_Map = None, Influence = True, extra_character = "zz?",Temperature =1 ,noise = 1e-8, compressed= True):

        self.compact_data = Compact_Training_Data.copy()

        self.Total_data_size = len(Compact_Training_Data)
        self.num_of_positions = np.shape(Compact_Training_Data)[1]

        self.alphabet_list = np.unique(self.compact_data)
        self.total_alphabet_size = len(self.alphabet_list)
        self.compress = compressed #<=if false, we will not use compressed
        self.Influence = Influence
        self.Temperature = Temperature
        self.noise = noise
        
        ###
        if self.compress:
            # we need to introduce extra character
            self.extra_character = self.alphabet_list[-1] *2
            #self.extra_character = extra_character
            self.Check_type()
        self.compressed_alphabet_size , self.conversion_dictionary = self.Compressed_dictionary()
        self.num_of_positions = len(self.compressed_alphabet_size)

        ###-- constructing the list of indices for each position ---
        self.Position_index = self.Position_index_finder()

   
        ### construct fundamental connectivity and weights and biases
        #self.Init_Boltzmann_Parameters()

        ### --
        ## -- unfurling the data and building lookup tables ---
        self.unfurled_data = self.Unfurled_dataset(self.compact_data)
                ## finding the unique data points and data multiplicity 
        ##### assigning the unique data and multiplicity to respective class properties
        self.Unique_datapoints, self.Data_multiplicity, self.unique_data_args = self.Unique_data_finder(self.unfurled_data)

        ###------- Partition Setup ----- 
        if type(Partition_Map)==type(None):
            ### assume all-all connected
            self.Raw_Partition_Map = np.ones(self.num_of_positions).reshape(1,-1)
            self.num_of_partitions = 1
        else:
            self.Raw_Partition_Map = Partition_Map.copy()
            self.num_of_partitions = len(Partition_Map)
        print("Number of Partitions Set=", self.num_of_partitions)

        self.Partition_Map_Setter(self.Raw_Partition_Map)

            
        self.data_lookup, self.full_data_Hash_values = self.build_lookup_structure(self.unfurled_data)

        #self.Init_Boltzmann_Parameters_ver2()
        self.Init_ScoreMatching_Scaled()


        ### Gauge fix
        for p in range(0, self.num_of_partitions):
            self.Bias[p] = self.Bias_Gauge_Fixing(p)

        ### 

    ### ----------- Helpers --------

    def build_lookup_structure(self, Data_Array):
        """Build optimized lookup structure once
        - For each UNIQUE data point as keys
        - the value will hold the probabilty (frequency/total_data_points)"""
        lookup = {}

        ## this contains hash values for all data points
        Hash_values = []

        length_of_data = len(Data_Array)
        for i, seq in enumerate(Data_Array):
            #hash(tuple(seq)) for memory efficiency
            # tuple(seq) is pretty inefficient
            key = hash(tuple(seq)) 
            Hash_values.append(key)
            if key in lookup:
                lookup[key] += 1
            else:
                lookup[key] = 1
        #return lookup, np.array(Hash_values)
        sorted_lookup_dict = dict(sorted(lookup.items()))
        Hash_values_sorted = np.array(list(sorted_lookup_dict.keys()))



        return sorted_lookup_dict, Hash_values_sorted
    

    def Hash_Value_given_data(self, Data_point):
        Value = hash(tuple(Data_point))
        return Value

    #def Hash_Values_batch(self, Data_points):
    #    """Batch hash computation for multiple data points"""
    #    # Method 1: Pure Python (good balance)
    #    return [hash(tuple(dp)) for dp in Data_points]
    
    def Hash_Values_batch(self, Data_points):
        """Fastest for pure Python"""
        return list(map(lambda dp: hash(tuple(dp)), Data_points))


    def Position_index_finder(self):
        """Constructs a dictionary of indices for each position in the one-hot encoded space"""
        Position_index = {}
        start_p = 0
        for p in range(0, len(self.compressed_alphabet_size)):
            end_p = start_p + self.compressed_alphabet_size[p]
            Position_index[p] = np.arange(start_p, end_p)
            start_p = end_p
        return Position_index
    

    def Unique_data_finder(self, Data):

        Data_Dict, datapoint_hash = self.build_lookup_structure(Data)
        #Unique_datapoint_hash = list(Data_Dict.keys())

        #Unique_datapoint_hash = np.array(list(Data_Dict.keys()))
        unique_datapoint_args = []
        for h in Data_Dict.keys():
            a_h = (np.argwhere(datapoint_hash==h)[0])
            unique_datapoint_args.append(a_h)
        unique_datapoint_args = np.array(unique_datapoint_args).reshape(-1)

        Multiplicity = np.array(list(Data_Dict.values()))
        Unique_datapoints  = Data[unique_datapoint_args]
        return Unique_datapoints, Multiplicity , unique_datapoint_args
        

    def Check_type(self):
        if self.extra_character not in self.alphabet_list:
            print("extra character=",self.extra_character )
            print("last alphabet =" , self.alphabet_list[-1])
            print("Extra character is okay.")

        else:
            ## last character is probably 0.
            m = np.random.randint(0, self.total_alphabet_size ,1)

            self.extra_character = self.alphabet_list[-1] +  self.alphabet_list[m]

            print("internal extra character was off")
            print("new extra character=",self.extra_character )
            print("last alphabet =" , self.alphabet_list[-1])
            self.Check_type()


    def Compressed_dictionary(self):
        ## will we compress it or not
        compression_status = self.compress

        dimension =[]
        conversion_dictionary ={}
        if compression_status:
            # we will compress it
            for i,c in enumerate(self.compact_data.T):
                unique_i = np.unique(c)
                dim_i = len(unique_i)+1
                unique_i = np.concatenate((unique_i, [self.extra_character]))
                conversion_dictionary[i] = unique_i
                dimension.append(dim_i)
        else:
            # we will not compress 
            Unique_set = self.alphabet_list
            for i in range(0, self.num_of_positions):
                unique_i = Unique_set
                dim_i = len(unique_i)
                conversion_dictionary[i] = Unique_set
                dimension.append(dim_i)

        return np.array(dimension), conversion_dictionary
    
    

    def Partition_Map_Setter(self, Raw_Partition_Map):
        print("..Setting Partition Map..")
        print("Number of partitions =", len(Raw_Partition_Map))

        """Sets the raw partition map
        - For each partition in the raw partitions, 
        constructs a list of indices pertaining to data in that partition in 1-hot encoded bases"""

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
        #self.Init_Boltzmann_Parameters_ver2()
        
        self.Init_ScoreMatching_Scaled()

        self.Hash_Value_by_partition, self.Lookup_table_by_partition = self.Partitioned_Data_Lookup_table()
        
        self.Mutant_weight = self.Mutant_representation_weights()



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

            try:
                arg_one = (np.argwhere(Dict_i==word_i).reshape(-1))
            except:
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
            start_index += dim_i
            Compact_state.append(word_i)

        return np.array(Compact_state).reshape(-1)

    def Unfurled_dataset(self, Data):
        """Takes any compact dataset and writes it in a one-hot encoded compressed basis
        - Compressed meaning all the words never seen at a position are mapped to a single position in one hot encoded vector for that word"""
        Full_set = []
        for d in Data:
            unf_d = self.One_hot_encoder(d)
            Full_set.append(unf_d)
        return np.array(Full_set)


    
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

             
    def Mutant_representation_weights(self):
        """ Gives weights of mutants for suppression
         - If a mutant is a such that it has been observed in the data , weight =1
         - If the mutant is such that it represents all unseen words at a given position (the filler), weight = Size of unseen words
         """

        Num_part = len(self.Raw_Partition_Map)
        Mu_Rep_wt = {}

        for Partition_index in range(0, Num_part):
            Map = self.Raw_Partition_Map[Partition_index]
            Args_one =np.argwhere(Map==1).reshape(-1)

            Compressed_alphabets = self.compressed_alphabet_size[Args_one]

            Indices = np.cumsum(Compressed_alphabets) -1

            Mu_weights = np.ones(np.sum(Compressed_alphabets))

            Remaining = self.total_alphabet_size -  Compressed_alphabets 

            Mu_weights[Indices] = Remaining
            Mu_Rep_wt[Partition_index] = Mu_weights

        return Mu_Rep_wt

    

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
    
    def Partitioned_Data_Lookup_table(self):
        Hash_values_by_partition = {}
        Lookup_tables_by_partition = {}
        # --- NEW ---
        Hash_values_by_partition_self = {}
        Lookup_tables_by_partition_self = {}
        # -----------

        for p in range(0, self.num_of_partitions):
            Data_in_partition_self, Data_in_partition_with_influence = self.Data_Partition(self.unfurled_data, p)

            Lookup_table_p, Hash_values_p = self.build_lookup_structure(Data_in_partition_with_influence)
            Hash_values_by_partition[p] = Hash_values_p
            Lookup_tables_by_partition[p] = Lookup_table_p

            # --- NEW: build self-only lookup ---
            Lookup_table_self_p, Hash_values_self_p = self.build_lookup_structure(Data_in_partition_self)
            Hash_values_by_partition_self[p] = Hash_values_self_p
            Lookup_tables_by_partition_self[p] = Lookup_table_self_p

        # Store both
        self.Hash_Value_by_partition_self = Hash_values_by_partition_self
        self.Lookup_table_by_partition_self = Lookup_tables_by_partition_self

        return Hash_values_by_partition, Lookup_tables_by_partition

    
    ######################### Different types of initializations

    def Init_Boltzmann_Parameters(self):
        """
         -Initialize with symmetric W_self
         - Layer mutant energy calculation can be made faster with BLAS routine for symmetric matrix
        """

        
        FC = self.Fundamental_Connectivity()
        self.fundamental_connectivity = FC
        
        Partitions = list(FC.keys())
        Weights = {}
        Bias = {}
        
        Data_mean = np.mean(self.unfurled_data, axis=0)
        
        for p in Partitions:
            Row_size, Col_size = np.shape(FC[p])
            Weight_p = np.random.normal(loc=0, scale=100*self.noise, size=(Row_size, Col_size))
            Weight_p *= FC[p]
            
            # ============================================================
            # Make W_self symmetric (3 lines)
            # ============================================================
            split_idx = (Col_size - Row_size) if (self.Influence and p > 0) else 0
            W_self = Weight_p[:, split_idx:]
            Weight_p[:, split_idx:] = (W_self + W_self.T) / 2
            # ============================================================
            
            p_indices = self.Partition_Map[p]
            #Data_mean_p = Data_mean[p_indices] * (-0.01)
            frequencies = np.clip(Data_mean[p_indices], 1e-5, 1)
            Bias_p = self.Temperature * np.log(frequencies) * 1e-5
            Bias_p += np.random.normal(loc=0, scale=self.noise, size=Row_size)
            
            
            Weights[p] = Weight_p
            Bias[p] = Bias_p
        
        self.Weights = Weights
        self.Bias = Bias
        
        self.Data_Partitioned = self.Data_part()
        self.partition_multiplicity_arg_dict = self.Data_arg_multiplicity_dictionary_constructor(self.unfurled_data)
    
    def Init_Boltzmann_Parameters_ver2(self):
        """
         -Initialize with symmetric W_self
         - Layer mutant energy calculation can be made faster with BLAS routine for symmetric matrix
        """
        print("semi-warm start. Matching only the frequencies")
        FC = self.Fundamental_Connectivity()
        self.fundamental_connectivity = FC
        
        Partitions = list(FC.keys())
        Weights = {}
        Bias = {}
        
        Data_mean = np.mean(self.unfurled_data, axis=0)
        
        for p in Partitions:
            Row_size, Col_size = np.shape(FC[p])
            p_indices = self.Partition_Map[p]
            
            # Initializing bias from frequencies
            frequencies = np.clip(Data_mean[p_indices], 1e-8, 1.0)
            Bias_p = self.Temperature * np.log(frequencies)
            
            # Measure typical bias magnitude
            bias_magnitude = np.mean(np.abs(Bias_p))
            
            # Initialize weights to be 20% of bias magnitude
            # This allows weights to have effect without dominating
            weight_std = 0.05 * bias_magnitude  # Adjust 0.2 to 0.1-0.5 as needed
            
            Weight_p = np.random.normal(loc=0, scale=weight_std, size=(Row_size, Col_size))
            Weight_p *= FC[p]
            
            # Make W_self symmetric
            split_idx = (Col_size - Row_size) if (self.Influence and p > 0) else 0
            W_self = Weight_p[:, split_idx:]
            Weight_p[:, split_idx:] = (W_self + W_self.T) / 2
            
            # Add tiny noise to bias for symmetry breaking
            Bias_p += np.random.normal(loc=0, scale=self.noise, size=Row_size)
            
            Weights[p] = Weight_p
            Bias[p] = Bias_p * 0.1
        
        self.Weights = Weights
        self.Bias = Bias



    def Init_ScoreMatching_Scaled(self):
        """
        FINAL RECOMMENDATION: Score matching / pseudolikelihood (same thing!)
        
        This is the analytical solution - no sklearn, no class balance issues
        """
        
        print("Initializing with score matching ...")


        
        Data = self.unfurled_data
        N = len(Data)
        Data_mean = np.mean(Data, axis=0)
        
        # Compute correlations
        Correlation = (Data.T @ Data) / N
        Connected_Corr = Correlation - np.outer(Data_mean, Data_mean)
        
        
        FC = self.Fundamental_Connectivity()
        self.fundamental_connectivity = FC
        
        Weights = {}
        Bias = {}
        
        for p in range(self.num_of_partitions):
            p_indices = self.Partition_Map[p]
            
            if self.Influence and p > 0:
                prev_indices = self.Partition_Map[p-1]
                all_indices = np.concatenate([prev_indices, p_indices])
            else:
                all_indices = p_indices
            
            C_block = Connected_Corr[np.ix_(p_indices, all_indices)]
            
            # Weights from correlations (pseudolikelihood analytical solution)
            Weight_p = -C_block / (1 + 0.1)
            Weight_p *= FC[p]
            
            # Bias from frequencies
            frequencies = np.clip(Data_mean[p_indices], 1e-8, 1.0)
            #Bias_p = self.Temperature * np.log(frequencies)
            Bias_p = -1*frequencies 
        
            
            # Make W_self symmetric
            Row_size, Col_size = Weight_p.shape
            split_idx = (Col_size - Row_size) if (self.Influence and p > 0) else 0
            W_self = Weight_p[:, split_idx:]
            Weight_p[:, split_idx:] = (W_self + W_self.T) / 2
            
            Bias_p += np.random.normal(loc=0, scale=self.noise, size=len(p_indices))
            
            Weights[p] = Weight_p
            Bias[p] = Bias_p
        
        self.Weights = Weights
        self.Bias = Bias


        

    #############
    def Bias_Gauge_Fixing(self, partition_index):
        Bias = self.Bias[partition_index].copy()
        partition_indices = np.argwhere(self.Raw_Partition_Map[partition_index]==1).reshape(-1)
        Alphabet_sizes= self.compressed_alphabet_size
        current_idx = 0

        Bias_new = np.zeros_like(Bias)
        for i in partition_indices:
            start_index = current_idx
            end_index = current_idx + Alphabet_sizes[i]
            Bias_i = Bias[start_index:end_index]
            Mean_i = np.mean(Bias_i)
            
            Bias_new[start_index:end_index] = Bias_i - Mean_i
            ## shift current to next position
            current_idx+=Alphabet_sizes[i]
        return Bias_new
    

    ### ----------- Energy Calculators -----------
    def Energy_Array_given_partition(self, Weight_part, Bias_part, Data_self, Data_interaction):
        """Calculates energy for an Array using partitioned weight/bias matrices"""

        Energy_all = np.sum(Data_self @ Weight_part * Data_interaction , axis = 1) + Data_self@Bias_part
        return Energy_all
    
    def Energy_Mutant_Array(self, Weight_part, Bias_part, Mutant_self_full, Mutant_interact_full):
        """Calculates energy for a Mutant Array using partitioned weight/bias matrices
        - This needs to have mutant arrays otherwise it doesn't work
        - Speed comes from the fact that the influencing piece for all Mutant is the same
        - Hence calculation can be fragmented into self energy + influence energy"""
        a,b = np.shape(Weight_part)
        W_self = Weight_part[:,(b-a):]
        W_interact = Weight_part[:, 0:(b-a)]
        #print(np.shape(W_self))
        #print(np.shape(W_interact))
        D_prev = Mutant_interact_full[0][0:(b-a)]

        Bias_Influence = (W_interact@D_prev) + Bias_part

        E_total = np.sum(Mutant_self_full @ W_self * Mutant_self_full, axis =1) + Mutant_self_full@Bias_Influence
        
        return E_total
        
    def Energy_Mutant_Array_Split(self, Weight_part, Bias_part,  Mutant_self_full, Mutant_interact_full):
        """
        Returns self-energy and interaction-energy separately.
        E_total = E_self + E_interact (+ bias, attributed to self)
        """
        a, b = np.shape(Weight_part)
        W_self     = Weight_part[:, (b-a):]
        W_interact = Weight_part[:, 0:(b-a)]
        D_prev     = Mutant_interact_full[0][0:(b-a)]

        # Self energy: quadratic term + bias (no influence from x_prev)
        E_self    = np.sum(Mutant_self_full @ W_self * Mutant_self_full, axis=1) \
                    + Mutant_self_full @ Bias_part

        # Interaction energy: how x_prev modulates x_curr
        E_interact = Mutant_self_full @ (W_interact @ D_prev)

        E_total = E_self + E_interact

        return E_self, E_interact, E_total


    def Energy_state_given_partition(self, Unfurled_Sequence, partition_index):
        
        """Calculates energy for a sequence using partitioned weight/bias matrices"""

        Data_self, Data_interaction = self.Data_Partition(Unfurled_Sequence, partition_index)

        Weight = self.Weights[partition_index]
        Bias = self.Bias[partition_index]
        
        Energy_Sequence = np.dot(Data_self @ Weight, Data_interaction) + np.dot(Bias, Data_self)
        
        return Energy_Sequence
    
    #####-------------------------------------
    
    def Data_Partition_args_and_multiplicity(self, Data, partition_index):
        Data_Self, Data_Interaction = self.Data_Partition(Data, partition_index)
        Data_interact_unique, Multiplicity_unique, Arg_unique = self.Unique_data_finder(Data_Interaction)
        return Data_interact_unique, Multiplicity_unique, Arg_unique
    
    def Data_arg_multiplicity_dictionary_constructor(self, Data):
        
        Full_arg_multiplicity_dict={}

        for p in range(0, len(self.Raw_Partition_Map)):
            p_dict ={}
            _, mult_p, arg_p = self.Data_Partition_args_and_multiplicity(Data,p)
            p_dict["Multiplicity"] = mult_p
            p_dict["Args"] = arg_p

            Full_arg_multiplicity_dict[p] = p_dict
            
        return Full_arg_multiplicity_dict
    
    
    ### --------- Constructing Mutants -----------

    def Single_Mutants_Generator(self, Unfurled_Sequence, partition_index, weighted_out_mutants=False):


        ## split the sequence into self and interaction components for this partition:
        Data_self, Data_interaction = self.Data_Partition(Unfurled_Sequence, partition_index)
        Diff = len(Data_interaction) - len(Data_self)

        Connectivity_partition = self.fundamental_connectivity[partition_index]

        Part_dof = len(Data_self)

        
        Args_zero =np.ravel(np.argwhere(Data_self==0))

        num_mutants = len(Args_zero)

        ### purely prev partition data:
        Data_prev_part = Data_interaction[:Diff]
        
        ## check to see 

        Connectivity_relevant = Connectivity_partition[:, Diff:]

        #Single_mutant_full_self = ((Connectivity_relevant * Data_self) + np.eye(Part_dof))

        ## we will remove all the copies of the datapoint from this mutant list
        ## this is done by selecting only 'Arg_zeros' mutants
        Single_mutant_self =  ((Connectivity_relevant * Data_self) + np.eye(Part_dof))[Args_zero]


        if weighted_out_mutants==False:
            Mutant_weight = np.ones(len(Single_mutant_self))

        else:
            Mutant_weight = self.Mutant_weight[partition_index][Args_zero]

        ### 
        D_prev = np.array([Data_prev_part for i in range(0, num_mutants)])

        Single_mutant_interaction = np.hstack([D_prev, Single_mutant_self])

        return Single_mutant_self, Single_mutant_interaction, Mutant_weight
    

        #####-------------------------------

    def Fast_Single_Mutants_Sorter_Partition_for_single_datapoint_OPTIMIZED(self, unfurled_sequence, partition_index, weighted_out_mutants=False):

        Hash_values = self.Hash_Value_by_partition[partition_index]
        Lookup_table = self.Lookup_table_by_partition[partition_index]

        # --- NEW: self-only structures ---
        Hash_values_self = self.Hash_Value_by_partition_self[partition_index]
        Lookup_table_self = self.Lookup_table_by_partition_self[partition_index]
        Lowest_hash_self = Hash_values_self[0]
        Highest_hash_self = Hash_values_self[-1]
        # ----------------------------------

        Lowest_hash = Hash_values[0]
        Highest_hash = Hash_values[-1]

        p_self, p_interact = self.Data_Partition(unfurled_sequence, partition_index)
        Single_mutants_self, Single_mutants_interaction, Mutant_weight = self.Single_Mutants_Generator(unfurled_sequence, partition_index, weighted_out_mutants)

        Single_mutants_self = np.vstack((p_self, Single_mutants_self))
        Single_mutants_interaction = np.vstack((p_interact, Single_mutants_interaction))
        Mutant_weight = np.concatenate(([1], Mutant_weight))

        mutant_hashes = self.Hash_Values_batch(Single_mutants_interaction)
        Hash_value_of_given_data = mutant_hashes[0]
        Data_property = {} 
        Data_property["Mutant_self"] = Single_mutants_self
        Data_property["Mutant_interaction"] = Single_mutants_interaction
        Data_property["Mutant_weight"] = Mutant_weight

        num_mutants = len(Single_mutants_interaction)
        Data_mutant_multiplicity = np.zeros(num_mutants, dtype=int)

        for i, hash_sm in enumerate(mutant_hashes):
            multiplicity = 0
            if Lowest_hash <= hash_sm <= Highest_hash:
                multiplicity = Lookup_table.get(hash_sm, 0)
            Data_mutant_multiplicity[i] = multiplicity

        # --- NEW: override row 0 (the data point itself) with self-multiplicity ---
        self_hash = hash(tuple(p_self))
        self_multiplicity = 0
        if Lowest_hash_self <= self_hash <= Highest_hash_self:
            self_multiplicity = Lookup_table_self.get(self_hash, 0)
        Data_mutant_multiplicity[0] = self_multiplicity
        # --------------------------------------------------------------------------

        Data_property["Mutant_multiplicity"] = Data_mutant_multiplicity

        return Hash_value_of_given_data, Data_property
        
    def Fast_Single_Mutants_Sorter_Partition_for_single_datapoint_OPTIMIZED_old(self, unfurled_sequence, partition_index, weighted_out_mutants=False):
        """
        Optimized version that pre-splits mutants into in-data and out-of-data
        This avoids masking operations later in the gradient calculation

        ### the very first row in Mutant self and Mutant interaction is the data itself
        """
        
        ## Get pre-computed data structures
        Hash_values = self.Hash_Value_by_partition[partition_index]
        Lookup_table = self.Lookup_table_by_partition[partition_index]
        
        ## Cache boundary values
        Lowest_hash = Hash_values[0]
        Highest_hash = Hash_values[-1]
        
        ## Properties of the data itself
        Data_property = {}

        p_self, p_interact = self.Data_Partition(unfurled_sequence, partition_index)

        Single_mutants_self, Single_mutants_interaction, Mutant_weight = self.Single_Mutants_Generator(unfurled_sequence, partition_index, weighted_out_mutants)

        ### here the data point is stacked on the single mutant lists
        Single_mutants_self = np.vstack((p_self, Single_mutants_self))
        Single_mutants_interaction = np.vstack((p_interact, Single_mutants_interaction))
        Mutant_weight= np.concatenate(([1], Mutant_weight))

        ## Vectorize hash computation
        mutant_hashes = self.Hash_Values_batch(Single_mutants_interaction)


        Hash_value_of_given_data =  mutant_hashes[0]
        

        Data_property["Mutant_self"] = Single_mutants_self
        Data_property["Mutant_interaction"] = Single_mutants_interaction
        Data_property["Mutant_weight"] = Mutant_weight

        ## Pre-allocate arrays with known size
        ### lets look at self rather than interaction
        num_mutants = len(Single_mutants_interaction)
        Data_mutant_multiplicity = np.zeros(num_mutants, dtype=int)
        
        ## Use direct dict lookup (O(1) average case)
        for i, hash_sm in enumerate(mutant_hashes):
            multiplicity = 0
            if Lowest_hash <= hash_sm <= Highest_hash:
                multiplicity = Lookup_table.get(hash_sm, 0)
            Data_mutant_multiplicity[i] = multiplicity

        Data_property["Mutant_multiplicity"] = Data_mutant_multiplicity
        
        return Hash_value_of_given_data, Data_property


    def Fast_Single_Mutants_Sorter_Partition_for_Training_Data_OPTIMIZED(self, partition_index, weighted_out_mutants=False):
        """
        Builds property dictionary for all training data
        Uses optimized single datapoint function with pre-split mutants
        """
        Training_Data = self.unfurled_data
        Data_properties_in_partition = {}

        for point in Training_Data:
            Hash_value_point, Data_properties_for_point = self.Fast_Single_Mutants_Sorter_Partition_for_single_datapoint_OPTIMIZED(point, partition_index, weighted_out_mutants=weighted_out_mutants)
            Data_properties_in_partition[Hash_value_point] = Data_properties_for_point
        
        Data_properties = {}
        Data_properties['Partition_index'] = partition_index

        Data_properties["Properties"] = Data_properties_in_partition 

        return Data_properties
    


    ####------------------------------------ Outflow -------------------------------------------------------
    def Layered_Outflow_Gradient_Single_Pass(self, Data_Properties, method , power = 1):
        """
        Combines Wself and Winteract gradients in a single loop 
        with a single energy call per data point.
        """
        partition_index = Data_Properties['Partition_index']
        Properties      = Data_Properties['Properties']
        Weight_part     = self.Weights[partition_index]
        Bias_part       = self.Bias[partition_index]
        Total_data_size = self.Total_data_size

        a, b = Weight_part.shape
        dof_prev = b - a

        W_grad       = np.zeros_like(Weight_part)
        W_inter_grad = np.zeros((a, dof_prev))
        Bias_grad    = np.zeros_like(Bias_part)
        Total_flow   = 0.0
        Norm = 0

        for hsh_val, prop in Properties.items():

            Mutant_self     = prop['Mutant_self']
            Mutant_interact = prop['Mutant_interaction']
            Mutant_mult     = prop['Mutant_multiplicity']

            p_i    = Mutant_mult[0] / Total_data_size
            x_prev = Mutant_interact[0][:dof_prev]

            Multiplicity = Mutant_mult[1:]

            p_i_power = p_i
            if method ==1:
                #Outflow:
                Factor = 1
                self.method = "Outflow"

            elif method == 2:
                #Gross Flow :
                Factor = 1*(Multiplicity>0) + 1
                self.method = "Gross Flow"

            elif method ==3:
                # MPF
                Factor = 1*(Multiplicity < 1)
                self.method = "MPF"

            elif method ==4:
                # Exponential
                Factor = 1
                p_i_power = np.exp(p_i) -1
                self.method = "Outflow Exponential"

            else:
                # Outflow with power
                Factor = 1
                self.method = "Outflow with power"
                #p_i_power = np.exp(p_i)-1
                p_i_power = p_i**power



            

            ## Just to prevent numerical nonesense
            Norm += p_i_power


            # --- Single energy call: self, interact, total ---
            E_self, _, E_total = self.Energy_Mutant_Array_Split(Weight_part, Bias_part, Mutant_self, Mutant_interact)

            # ---- W_self gradient: uses E_self, smaller matmul ----
            E_seq_self     = E_self[0]
            E_mutants_self = E_self[1:]

            Exp_Delta_self = np.exp((E_seq_self - E_mutants_self) / (2 * self.Temperature)) * p_i_power * Factor
            Sum_flow_self  = np.sum(Exp_Delta_self)

            Scaled_self    = Mutant_self[1:] * Exp_Delta_self.reshape(-1, 1)

            # (a, n_mut) @ (n_mut, a) -> (a, a) — self block only, no mask needed
            W_self_grad_i  = Scaled_self.T @ Mutant_self[1:]
            W_self_grad_i -= np.outer(Mutant_self[0], Mutant_self[0]) * Sum_flow_self

            Bias_grad_i    = np.sum(Scaled_self, axis=0) - Mutant_self[0] * Sum_flow_self

            # Write directly into W_self block
            W_grad[:, dof_prev:] += W_self_grad_i 
            Bias_grad            += Bias_grad_i

            # ---- W_interact gradient: uses E_total, already correct ----
            if dof_prev > 0:
                E_seq_total     = E_total[0]
                E_mutants_total = E_total[1:]

                Exp_Delta_inter  = np.exp((E_seq_total - E_mutants_total) / (2 * self.Temperature)) * p_i_power * Factor
                Sum_flow_inter   = np.sum(Exp_Delta_inter)

                Scaled_xcurr_sum = Mutant_self[1:].T @ Exp_Delta_inter  
                W_inter_grad_i   = np.outer(Scaled_xcurr_sum, x_prev)
                W_inter_grad_i  -= np.outer(Mutant_self[0], x_prev) * Sum_flow_inter

                W_inter_grad += W_inter_grad_i

            Total_flow += Sum_flow_self


        # Apply connectivity mask to W_self block only
        W_grad[:, dof_prev:] *= self.fundamental_connectivity[partition_index][:, dof_prev:]
        

        # Embed W_inter into full gradient matrix
        W_grad[:, :dof_prev] = W_inter_grad

        ### Normalization
        W_grad *=(1/Norm)
        Bias_grad *= (1/Norm)

        return W_grad, Bias_grad, Total_flow
    

    def Exact_Min_Outflow_Single_Iteration_Parameter_Update(self, Data_Properties, epsilon, method = 1, power = 1):
        """
        Single iteration parameter update with FIXED BUGS
        """
        partition_index = Data_Properties['Partition_index']
        Weight_part = self.Weights[partition_index]
        Bias_part = self.Bias[partition_index]


        # Compute gradients
        Weight_Gradient_contribution, Bias_Gradient_contribution, Total_Outflow = self.Layered_Outflow_Gradient_Single_Pass(Data_Properties, method=method, power = power)
        
        # Update parameters
        Weight_part += epsilon * Weight_Gradient_contribution
        Bias_part += epsilon * Bias_Gradient_contribution

        # Assign to appropriate carriers
        self.Weights[partition_index] = Weight_part
        self.Bias[partition_index] = Bias_part

        #Gauge fixing in Bias. We will fix the mean at each position
        self.Bias[partition_index] = self.Bias_Gauge_Fixing(partition_index)

        return Total_Outflow

    

        
