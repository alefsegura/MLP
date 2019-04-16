import numpy as np
import pandas as pd
import random
from collections import namedtuple
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


"""
    processing: Function to transform divides the dataset in train and test and transform the Y values in binary, OneHotEnconding
"""
def processing(X, y, percentage):
    
    # Normalizing 
    x = StandardScaler().fit_transform(X) 

    # Labelizing 
    y = y.reshape(len(y), 1)
    y = OneHotEncoder(sparse=False).fit_transform(y)
        
    # Computing the lenght of dataset
    lenght = X.shape[0]

    #split dataset into train and test.
    x_train = x[0:int(percentage*lenght), :]
    y_train = y[0:int(percentage*lenght), :]

    x_test = x[int(percentage*lenght):, :]
    y_test = y[int(percentage*lenght):, :]
        
    #creating an alias to train and test set.
    dataset = namedtuple('datset', 'X Y')
    train = dataset(X=x_train, Y=y_train)
    test = dataset(X=x_test, Y=y_test)

    return train, test

"""
    sigmoid: Function to apply the sigmoid function, used in the backpropagation step.
""" 
def sigmoid(x):
    return 1/(1+np.exp(-x))

"""
    mlp_forward: Function responsible for the forward step, that applies the actual weight values on the net.
"""
def mlp_forward(x, hidden_weights, output_weights):

    f_net_h = []
    #apllying the weights on the hidden units.
    for i in range(len(hidden_weights)):
        #if is the first hidden unit.
        if i == 0:
            net = np.matmul(x,hidden_weights[i][:,0:len(x)].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        #if is the second or more hidden unit
        else:
            net = np.matmul(f_net_h[i-1],hidden_weights[i][:,0:len(f_net_h[i-1])].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        
        #store f_net of hidden layers.
        f_net_h.append(f_net) 

    #computing the net function to the output layer.
    net = np.matmul(f_net_h[len(f_net_h)-1],output_weights[:,0:len(f_net_h[len(f_net_h)-1])].transpose()) + output_weights[:,-1]
        
    f_net_o = sigmoid(net)
    
    return f_net_o, f_net_h

"""
    mlp_backward: Function responsible for the backpropagation step, which corresponds to the updating of weights.
"""
def mlp_backward(dataset, j, hidden_weights, output_weights, f_net_o, f_net_h, eta, hidden_units, alpha, momentum_h, momentum_o, n_classes):

    x = dataset.X[j,:]
    y = dataset.Y[j,:]
    
    # Measuring the error
    error = y - f_net_o
    delta_o = error*f_net_o*(1-f_net_o)
    
    # Computing the delta for the hidden units
    delta_h = []
    for i in range(len(hidden_units)-1, -1, -1):

        if(i == len(hidden_units)-1):
            w_o = output_weights[: ,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta_o, w_o))
        else:
            w_o = hidden_weights[i+1][:,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta, w_o))

        delta_h.insert(0,delta)
    
    # Computing the delta and updating weights for the output layer
    delta_o = delta_o[:, np.newaxis]
    f_net_aux = np.concatenate((f_net_h[len(hidden_units)-1],np.ones(1)))[np.newaxis, :]
    output_weights = output_weights - -2*eta*np.matmul(delta_o, f_net_aux) + momentum_o
    momentum_o = - -2*eta*np.matmul(delta_o, f_net_aux)
    
    # Cpdating the weights for the hidden layers
    for i in range(len(hidden_units)-1, -1, -1):
        delta = delta_h[i][:, np.newaxis]
        f_net_aux = np.concatenate((f_net_h[i],np.ones(1)))[np.newaxis, :]    

        if i == 0:
            x_aux = np.concatenate((x,np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, x_aux) + momentum_h[i]
            momentum_h[i] = - -2*eta*np.matmul(delta, x_aux)
        else:
            f_net_aux = np.concatenate((f_net_h[i-1],np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, f_net_aux) + momentum_h[i]
            momentum_h[i] = - -2*eta*np.matmul(delta, f_net_aux)

    # Measuring the error
    error = sum(error*error)

    # Return the updated weights, the new error and the momentum parameters.
    return hidden_weights, output_weights, error, momentum_h, momentum_o

"""
    testing: Function responsible to realize the tests for the classification and regression methods for different datasets.
"""
def testing(train, test, hidden_weights, output_weights):
    counter = 0

    for i in range(test.X.shape[0]):
        y_hat, q = mlp_forward(test.X[i,:], hidden_weights, output_weights)
        y_hat = np.argmax(y_hat)
        y = np.argmax(test.Y[i,:])
        if y == y_hat:
            counter += 1

    return counter/test.X.shape[0]

"""
    MLP: function responsible to initialize weights, check conditions to construct net.
"""
def MLP(X, y, hidden_units, epochs, eta, alpha, data_ratio):
    n_classes = len(np.unique(y))
    hidden_layers = len(hidden_units)
    
    # Acquiring the train and test set
    train, test = processing(X, y, data_ratio)

    # Initializing the weights of the hidden layers
    momentum_o = 0
    momentum_h = []
    hidden_weights = []
    for i in range(hidden_layers):
        if(i == 0):
            aux = np.zeros((hidden_units[i], X.shape[1] + 1))
        else:
            aux = np.zeros((hidden_units[i], hidden_units[i-1] + 1))
        hidden_weights.append(aux)
        momentum_h.append(aux)
  
    # Filling the hidden layers weight values with a normal distribution between -1 and 1
    for i in range(hidden_layers):
        for j in range(hidden_units[i]):
            if(i == 0):
                for k in range(X.shape[1] + 1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)
            else:
                for k in range(hidden_units[i-1]+1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)

    # Initializing and filling the weights values of output layer
    output_weights = np.zeros((n_classes, hidden_units[len(hidden_units)-1]+1))

    for i in range(n_classes):
        for j in range(hidden_units[hidden_layers-1]+1):
            output_weights[i][j] = random.uniform(-1, 1)

    epoch = 0
    for epoch in range(epochs):
        print
        sum_errors = 0
        for i in range(train.X.shape[0]):
            # Forward
            f_net_o, f_net_h = mlp_forward(train.X[i, :], hidden_weights, output_weights)
            # Backward hidden_weights, output_weights, error
            hidden_weights, output_weights, error, momentum_h, momentum_o = mlp_backward(train, i, hidden_weights, output_weights, f_net_o, f_net_h, eta, hidden_units, alpha, momentum_h, momentum_o, n_classes)
            sum_errors += error
        epoch += 1

    # Evaluating    
    return testing(train, test, hidden_weights, output_weights)