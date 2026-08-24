import numpy as np
import itertools

def Potts_Parameter_Creator(Number_of_sites, Number_of_Alphabets, High_value, fraction_high, noise = 1e-4):

    """Initialized a random potts model with some high values sprinkled to provide structure
    Parameters:
    - Number_of_sites = (integer) Number of sites in the problem
    - Number_of_Alphabets = (integer) NUmber of alphabets that each site can take (internal deg of freedom)
    - High value = (float) The scale (st dev) of gaussian that produces high value to sprinkle in the potts model
    - fraction_ high = (0<=float <=1) the fraction of parameters on which we sprinkle high value
    - noise = Noise threshold for the rest of the parameter
    
    Output:
    Weight Matrix, Bias Vector
        """

    Total_dof = Number_of_sites * Number_of_Alphabets
    High_value = np.abs(High_value)

    FC = np.ones(shape=(Total_dof, Total_dof))
    for s in range(0, Total_dof, Number_of_Alphabets):
        start = s
        end = s+ Number_of_Alphabets
        FC[start:end, :][:, start:end] = 0


    W = np.random.normal(loc = 0, scale = noise, size=(Total_dof, Total_dof)) * FC
    
    B = np.random.normal(loc = 0, scale = noise, size=Total_dof)

    ## change weights (sprinkle high values to give structure)
    Args_non_zero_weight = np.argwhere(np.triu(FC)!=0)

    print("Number of High values=", len(Args_non_zero_weight))
    Num_select = int(len(Args_non_zero_weight) * fraction_high)

    
    Args_arg_weight_weight = np.random.choice(range(0, len(Args_non_zero_weight)), size = Num_select, replace=False)
    
    Args_selected_weight = Args_non_zero_weight[Args_arg_weight_weight]

    for a in Args_selected_weight:
        #r,c = a
        W[a[0],a[1]] = (np.random.normal(loc =0, scale= High_value ,size = 1))[0]

    W = (W+W.T)/2
    W *=FC

    Args_selected_bias = np.ravel(np.random.choice(np.arange(0, Total_dof), size = int(Total_dof * fraction_high)))

    B[Args_selected_bias] = np.random.normal(loc =0, scale= High_value ,size = len(Args_selected_bias))

    #Partition_Map = np.ones(Number_of_sites)
    #Partition_Map = Partition_Map.reshape(1,Number_of_sites)
    #Args_non_zero

    return W, B 



def One_hot_encoder(Compact_state, Num_letter):
    """For any given state, it returns a one hot encoded state
    - unseen words at any position are mapped to unknown"""
    Dict = range(0, Num_letter)

    unfurled=[]
    for elem in Compact_state:
        subvect = np.zeros(Num_letter)
        subvect[elem] = 1
        unfurled.append(subvect)
    unfurled_state = np.concatenate(unfurled)
    return unfurled_state

def Potts_Energy_of_State_Array(State_Array, Weight, Bias):
    """Calculates energy for an one hot encoded Array using partitioned weight/bias matrices"""
    Energy_all = np.sum(State_Array @ Weight * State_Array , axis = 1) + State_Array@Bias
    return Energy_all

def Potts_Probability_of_State_Array(Energy_Array, Temperature):
    """Given an array of Energy for all states, this computes the Boltzmann Probability at a given Temperature"""
    E_max = np.max(Energy_Array)
    Likelihood = np.exp(-1*(Energy_Array-E_max)/Temperature)
    Partition_Function = np.sum(Likelihood)
    Probability = Likelihood/Partition_Function

    return Probability


def Construct_States(Num_site, Num_letter):
    Alphabets = np.arange(0, Num_letter)
    Iterable = [Alphabets]*  Num_site
    print("Constructing all states")
    States = list(itertools.product(*Iterable))
    Num_states_constructed = len(States)
    States = np.array(States)
    print("Number of States from Construction=", Num_states_constructed)
    return np.array(States)


def Sample_From_Distribution(num_sampled, Full_States_library, Probability_Distribution):

    """Sample a given number of individuals according to the Potts model's Probability"""
    
    Picked = {}
    Num_states = len(Probability_Distribution)
    Indices_Picked = np.random.choice(range(0,Num_states), replace = True, p=Probability_Distribution, size = num_sampled)


    Chosen_States_Compact = Full_States_library[Indices_Picked]
    
    Chosen_State_Probability = Probability_Distribution[Indices_Picked]

    ### ---- Lets find the unique states ----
    Unique_Chosen, Counts_Chosen = np.unique(Chosen_States_Compact, axis = 0, return_counts=True)

    Unique_State_empirical_Probability = Counts_Chosen/np.sum(Counts_Chosen)

    Unique_args = []
    for u in Unique_Chosen:
        Arg = np.argmax(np.sum(Chosen_States_Compact == u, axis =1))
        Unique_args.append(Arg)
    Unique_args= np.array(Unique_args)

    Unique_State_GT_Probability = Chosen_State_Probability[Unique_args]

    Picked["Indices_Picked"] = Indices_Picked
    Picked["Chosen_Compact_States"] = Chosen_States_Compact 
    Picked['Unique_indices'] = Indices_Picked[Unique_args]
    Picked["Unique_chosen_compact_States"] = Unique_Chosen
    Picked["Ground_Truth_Probability_for_unique"] = Unique_State_GT_Probability
    Picked["Empirical_Probability_for_unique"] = Unique_State_empirical_Probability

    return Picked
