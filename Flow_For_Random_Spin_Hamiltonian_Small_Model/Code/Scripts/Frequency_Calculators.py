import numpy as np

####  helpers for frequency calculation:

### Here is a script for calculating the correlations in a given dataset:

def Singlet_Frequency(One_hot_encoded, Alphabet_size):
    """ Input - 
        One_hot_encoded : Takes in One hot encoded dataset
        Alphabet_size : Number of alphabets that could go in any position (internal deg of freedom)

        Output- singlet frequencies in the given dataset
        f_(i)(a)= Freq of seeing amino acids a at positions i
        """

    Mean_array = np.mean(One_hot_encoded, axis = 0)
    return Mean_array


def Pairwise_Frequency(One_hot_encoded, Alphabet_size):
    """ Input - 
        One_hot_encoded : Takes in One hot encoded dataset
        Alphabet_size : Number of alphabets that could go in any position (internal deg of freedom)
        
        Output- 
        Pairwise correlations and Correlation matrix for the given dataset 
        f_(ij)(a, b) = f_(ij)(a,b) """
    
    num_seq, length_seq = np.shape(One_hot_encoded)
    num_position = length_seq//Alphabet_size
    frequencies = []
    Matrix = np.zeros([length_seq,length_seq])
    for i in range(0, num_position):
        for j in range(i+1, num_position):
        
            #freq_ij = np.zeros((Alphabet_size, Alphabet_size, Alphabet_size))
            for s_i in range(i*Alphabet_size, (i+1) * Alphabet_size):
                for s_j in range(j*Alphabet_size, (j+1) * Alphabet_size):
                    vec_i = One_hot_encoded[:,s_i]
                    vec_j = One_hot_encoded[:,s_j]
                    Stat = np.sum(vec_i * vec_j )/ num_seq
                    
                    frequencies.append(Stat)
                    Matrix[s_i, s_j] = Stat
                    Matrix[s_j, s_i] = Stat

    return np.array(frequencies), Matrix



def Pairwise_Correlation(One_hot_encoded, Alphabet_size):
    """ Input - 
        One_hot_encoded : Takes in One hot encoded dataset
        Alphabet_size : Number of alphabets that could go in any position (internal deg of freedom)

        Output- 
        Pairwise correlations and Correlation matrix for the given dataset 
        Corr_(ij)(a, b) = f_(ij)(a,b) - f_i(a)f_i(b)"""
    
    Single_freq = Singlet_Frequency(One_hot_encoded, Alphabet_size)
    num_seq, length_seq = np.shape(One_hot_encoded)
    num_position = length_seq//Alphabet_size
    Correlations = []
    Matrix = np.zeros([length_seq,length_seq])
    for i in range(0, num_position):
        for j in range(i+1, num_position):
            #freq_ij = np.zeros((Alphabet_size, Alphabet_size, Alphabet_size))
            for s_i in range(i*Alphabet_size, (i+1) * Alphabet_size):
                for s_j in range(j*Alphabet_size, (j+1) * Alphabet_size):
                   
                    vec_i = One_hot_encoded[:,s_i]
                    vec_j = One_hot_encoded[:,s_j]
                    Stat = (np.sum(vec_i * vec_j )/ num_seq) - (Single_freq[s_i] * Single_freq[s_j])
                    
                    Correlations.append(Stat)
                    Matrix[s_i, s_j] = Stat
                    Matrix[s_j, s_i] = Stat

    return np.array(Correlations), Matrix


def Triplet_Frequency(One_hot_encoded, Alphabet_size):
    """ 
    Input - 
        One_hot_encoded : Takes in One hot encoded dataset
        Alphabet_size : Number of alphabets that could go in any position (internal deg of freedom)

        Output- triplet frequencies in the given dataset
        f_(ijk)(a,b,c)= Freq of seeing amino acids a,b,c at positions i,j,k respectively

        """

    num_seq, length_seq = np.shape(One_hot_encoded)
    num_position = length_seq//Alphabet_size
    frequencies = []
    for i in range(0, num_position):
        for j in range(i+1, num_position):
            for k in range(j+1, num_position):
                freq_ijk = np.zeros((Alphabet_size, Alphabet_size, Alphabet_size))
                for s_i in range(i*Alphabet_size, (i+1) * Alphabet_size):
                    for s_j in range(j*Alphabet_size, (j+1) * Alphabet_size):
                        for s_k in range(k*Alphabet_size, (k+1) * Alphabet_size):

                            vec_i = One_hot_encoded[:,s_i]
                            vec_j = One_hot_encoded[:,s_j]
                            vec_k = One_hot_encoded[:,s_k]

                            Stat = np.sum(vec_i * vec_j * vec_k)/ num_seq

                            frequencies.append(Stat)

    return frequencies





def Triplet_Correlation(One_hot_encoded, Alphabet_size):

    """ Input - 
        One_hot_encoded : Takes in One hot encoded dataset
        Alphabet_size : Number of alphabets that could go in any position (internal deg of freedom)

        Output- 
        triplet correlations in the given dataset
        """
    
    Single_freq = Singlet_Frequency(One_hot_encoded, Alphabet_size)
    _, PW_corr_matrix = Pairwise_Correlation(One_hot_encoded, Alphabet_size)

    num_seq, length_seq = np.shape(One_hot_encoded)
    num_position = length_seq//Alphabet_size
    frequencies = []
    for i in range(0, num_position):
        for j in range(i+1, num_position):
            for k in range(j+1, num_position):
                freq_ijk = np.zeros((Alphabet_size, Alphabet_size, Alphabet_size))
                for s_i in range(i*Alphabet_size, (i+1) * Alphabet_size):
                    for s_j in range(j*Alphabet_size, (j+1) * Alphabet_size):
                        for s_k in range(k*Alphabet_size, (k+1) * Alphabet_size):
                            vec_i = One_hot_encoded[:,s_i]
                            vec_j = One_hot_encoded[:,s_j]
                            vec_k = One_hot_encoded[:,s_k]
                            Singlet_contrib = Single_freq[s_i] * Single_freq[s_j] * Single_freq[s_k]
                            Pairwise_contrib = PW_corr_matrix[s_i,s_j] * Single_freq[s_k] + PW_corr_matrix[s_j,s_k] * Single_freq[s_i] + PW_corr_matrix[s_i,s_k] * Single_freq[s_j]  
                            Stat = (np.sum(vec_i * vec_j * vec_k)/ num_seq) - Singlet_contrib - Pairwise_contrib

                            frequencies.append(Stat)

    return frequencies