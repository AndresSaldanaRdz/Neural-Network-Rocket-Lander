import math
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, n_type, num_weights=0, learning_rate=0.1):
        self.n_type = n_type
        self.prev_weights = np.ones(num_weights)
        self.weights = np.random.random_sample(num_weights) #array of random weights
        self.activation_value = 0
        self.learning_rate = learning_rate
        self.current_values = np.ones(num_weights) #temporary values as the move across the nn
        
    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))
        
    def value_x_weight(self, value):
        "multiply a recieved value by each weight path that a neuron has, and hold that value"
        self.current_values = self.weights * value
            
    def value_x_activation(self, sum_weights):
        """activation function, recieves the sum of weights, activitaes it, then stores the value"""
        self.activation_value = self.sigmoid(sum_weights)
            
    def update_weight_hl(self, weight_num, output, target_out):
        """perfrm gradient decent to calcualte the next weight, return certain part of the chain as we can reuse it in the next later"""
        # new weight = old weight * learning rate  * partial derivative cost * partial derivative sigmoid: output * partial derivative sum: hidden neuron activation value
        self.prev_weights[weight_num] = self.weights[weight_num]
        self.weights[weight_num] = self.weights[weight_num] + self.learning_rate * (target_out - output) * (output * (1 - output)) * (self.activation_value)
        
    def update_weight_il(self, weight_num, output_list, nn_des_out, previous_w, hl_act, nn_input):
        """perfrm gradient decent to calcualte the next weight, """
        c = 0
        for output, des_out, prev_w, nni  in zip(output_list, nn_des_out, previous_w, nn_input):
            c += output * (1 - output) * (des_out - output) * prev_w * hl_act * (1 - hl_act) * nni
        self.weights[weight_num] = self.weights[weight_num] + self.learning_rate * (c)

    def __str__(self):
        return f"{self.n_type} | Weights: {self.weights} | Activation_V: {self.activation_value}"
        
def print_all(il,hl,ol):
    print(" ")
    for i in il:
        print(i)
    for i in hl:
        print(i)
    for i in ol:
        print(i)
    print(" ")
        
def create_nn(num_of_neurons_il = 2, num_of_neurons_hl = 3, num_of_neurons_ol = 2, learning_rate=0.3):
    """function to create a neural network with any # of neoron on each layer, input layer (il)
    # of neurons in  hidden layer (hl) = number of weights (il)<->(hl)
    # of neurons in the output layer (ol) = number of weights (hl)<->(ol)"""
    
    input_layer, hidden_layer, output_layer = [], [], []
    
    for i in range(num_of_neurons_il): # create the neurons and weights
        input_layer.append(Neuron(n_type = "input", num_weights=num_of_neurons_hl, learning_rate = learning_rate))

    for i in range(num_of_neurons_hl): # create the neurons and weights
        hidden_layer.append(Neuron(n_type = "hidden", num_weights=num_of_neurons_ol, learning_rate = learning_rate))

    for i in range(num_of_neurons_ol): # create the neurons
        output_layer.append(Neuron(n_type = "output", learning_rate = learning_rate))
    
    return input_layer, hidden_layer, output_layer

def forward_prop(nn_input, input_layer, hidden_layer, output_layer):
    for i, j in enumerate(nn_input):
        input_layer[i].value_x_weight(j)
        
    for hl_neuron, i in zip(hidden_layer, range(len(hidden_layer))): # I THINK THIS COULD HAVE BEEN DONE WITH AN ENUMERATE
        sum_of_values = 0
        for j in input_layer:
            sum_of_values += j.current_values[i] #sum of the (il)weights that will go into the (hl)neuron 
        hl_neuron.value_x_activation(sum_of_values) #activiation
        hl_neuron.value_x_weight(hl_neuron.activation_value) #multiply the activation result by the next weights (hl)
    
    for ol_neuron, i in zip(output_layer, range(len(output_layer))):
        sum_of_values = 0
        for j in hidden_layer:
            sum_of_values += j.current_values[i] #sum of the weights that will go into the neuron
        ol_neuron.value_x_activation(sum_of_values) #activation function "linear"

    return input_layer, hidden_layer, output_layer
    
def backward_prop(nn_input, nn_des_out, input_layer, hidden_layer, output_layer):

    output_list = []

    for i, ol_neuron in enumerate(output_layer): # "i" matches the corresponding weights that go into that neuron, 
        output = ol_neuron.activation_value
        output_list.append(output)
        for hl_neuron in hidden_layer:
            hl_neuron.update_weight_hl(i, output, nn_des_out.iloc[i])

    w_count = 0 #creo que este es el que esta bien
    for i, il_neuron in enumerate(input_layer):
        for j, hl_neuron in enumerate(hidden_layer):
            il_neuron.update_weight_il(j, output_list, nn_des_out, hl_neuron.prev_weights, hl_neuron.activation_value, nn_input)
            w_count+=1
    
    return input_layer, hidden_layer, output_layer

def calculate_accuracy(test_scaled, samples_num, margin, il, hl, ol, cross_val):
    """ function to calculate the accuracy of the model with random test dataset samples, this is the stopping criteria, evaluates accuracy by neuron and total.
    By total accuracy we evaluate is both outputs of each neuron compared to the target have a minimal error"""
    accuracy_list1, accuracy_list2, accuracy_list3 = [], [], []
    for i in range(cross_val):

        random_sample = test_scaled.sample(n=samples_num)
        accuracy_count1, accuracy_count2, accuracy_count3 = 0, 0, 0

        for index, row in random_sample.iterrows():
            nn_input, desired_out = row[0:2], row[2:4]
            il, hl, ol = forward_prop(nn_input, il, hl, ol)

            if (abs((desired_out.iloc[0] - ol[0].activation_value)) + abs((desired_out.iloc[1] - ol[1].activation_value)))/ 2 <= margin: # total, if both errors are minimal
                accuracy_count1 +=1

            if abs((desired_out.iloc[0] - ol[0].activation_value)) <= margin: # output neuron 1
                accuracy_count2 +=1

            if abs((desired_out.iloc[1] - ol[1].activation_value)) <= margin: # output neuron 2
                accuracy_count3 +=1

        accuracy_list1.append((accuracy_count1 / samples_num) * 100)
        accuracy_list2.append((accuracy_count2 / samples_num) * 100)
        accuracy_list3.append((accuracy_count3 / samples_num) * 100)
            
    return [sum(accuracy_list1) / cross_val, sum(accuracy_list2) / cross_val, sum(accuracy_list3) / cross_val]

def prep_data(file):   
    column_names = ['X_dtt', 'Y_dtt', 'X_vel', 'Y_vel'] 
    #dtt = distance to target (distance in pixels to the target), vel = velocity (Pixels per second)
    df = pd.read_csv(file, names=column_names)

    test_size = 0.08 #Splitting into training and test sets
    n_samples = len(df)

    test_indices = np.random.choice(df.index, size=int(n_samples * test_size), replace=False) #cretae random row indices to be able to split the dat ainto training and testing

    test = df.loc[test_indices] # Using the test indices to select rows for the test set
    train = df.drop(test_indices) # Use the remaining indices for the training set

    #Scaling the data
    # Calculate the minimum and maximum values for each column in the training set and the test set 
    test_min_values = test.min(axis=0)
    test_max_values = test.max(axis=0)

    train_min_values = train.min(axis=0)
    train_max_values = train.max(axis=0)

    # Perform Min-Max scaling on the training and testing sets
    test_scaled = (test - test_min_values) / (test_max_values - test_min_values)
    train_scaled = (train - train_min_values) / (train_max_values - train_min_values)

    return train_scaled, test_scaled

def train_nn(epochs, train_scaled, test_scaled, il, hl, ol, cut_off):
    for i in range(epochs): #epochs
        for index, row in train_scaled.iterrows():
            nn_input, desired_out = row[0:2], row[2:4]
            il, hl, ol = forward_prop(nn_input, il, hl, ol)
            il, hl, ol = backward_prop(nn_input, desired_out, il, hl, ol)

        acc = calculate_accuracy(test_scaled, 100, .05, il, hl, ol, 10)
        print(f"Epoch: {i} | Accuracy = Total: {acc[0]} Neuron 1: {acc[1]} Neuron 2: {acc[2]}")
        if float(acc[0]) > cut_off: # set the accuracy you want to achieve
            print("- - - - - Training Finished - - - - -")
            break

    return il, hl, ol

def create_weights_array(il,hl,ol):
    weights_list = []
    for i in il,hl,ol:
        for j in i:
            weights_list.append(j.weights)
            
    with open("weights.txt", 'w') as file:
        for array in weights_list:
            # Convert each array to a string and write it to the file
            array_str = ' '.join(map(str, array))
            file.write(array_str + '\n')

def load_weights_array(file, il, hl, ol):
    weights = []
    with open(file, 'r') as file:
        for line in file:
            # Read each line from the file, split it into values, and convert them back to an array
            array = np.array(list(map(float, line.strip().split())))
            weights.append(array)

    counter = 0
    for i in il,hl,ol:
        for j in i:
            j.weights = weights[counter]
            counter += 1

    return il, hl, ol

