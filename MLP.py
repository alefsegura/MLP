import numpy as np
import random
from collections import namedtuple

class MLP:
    
    def __init__(self, hidden_units, n_classes, learning_rate, delta_error):
        self.hidden_units = hidden_units
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.delta_error = delta_error
    
    def train_test_split(self, dataset, train_size):
        lenght = dataset.X.shape[0]
        x_train = dataset.X[0:int(train_size*lenght), :]
        y_train = dataset.Y[0:int(train_size*lenght), :]
        x_test = dataset.X[int(train_size*lenght):, :]
        y_test = dataset.Y[int(train_size*lenght):, :]
        dataset = namedtuple('datset', 'X Y')
        train = dataset(X=x_train, Y=y_train)
        test = dataset(X=x_test, Y=y_test)
        return train, test

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
       
    def forward(self, x, hidden_weights, output_weights):
        f_net_h = []
        # Pesos das camadas ocultas
        for i in range(len(hidden_weights)):
            # para a camada de entrada
            if i == 0:
                net = np.matmul(x, hidden_weights[i][:,0:len(x)].transpose()) + hidden_weights[i][:,-1]
                f_net = self.sigmoid(net)
            # demais camadas
            else:
                net = np.matmul(f_net_h[i-1], hidden_weights[i][:,0:len(f_net_h[i-1])].transpose()) + hidden_weights[i][:,-1]
                f_net = self.sigmoid(net)
            f_net_h.append(f_net) 

        # Camada de saída
        net = np.matmul(f_net_h[len(f_net_h)-1],output_weights[:,0:len(f_net_h[len(f_net_h)-1])].transpose()) + output_weights[:,-1]  
        f_net_o = self.sigmoid(net)

        return f_net_o, f_net_h

    def backward(self, dataset, j, hidden_weights, output_weights, f_net_o, f_net_h, eta, hidden_units, momentum_h, momentum_o, n_classes):
        x = dataset.X[j,:]
        y = dataset.Y[j,:]
        error = y - f_net_o
        delta_o = error*f_net_o*(1-f_net_o)

        # Delta para os parâmetros das camadas escondidas
        delta_h = []
        for i in range(len(hidden_units)-1, -1, -1):
            if i == len(hidden_units)-1:
                w_o = output_weights[: ,0:hidden_units[i]]
                delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta_o, w_o))
            else:
                w_o = hidden_weights[i+1][:,0:hidden_units[i]]
                delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta, w_o))

            delta_h.insert(0,delta)

        # Delta da camada de saída
        delta_o = delta_o[:, np.newaxis]
        f_net_aux = np.concatenate((f_net_h[len(hidden_units)-1],np.ones(1)))[np.newaxis, :]
        output_weights = output_weights - -2*eta*np.matmul(delta_o, f_net_aux) + momentum_o
        momentum_o = -(-2)*eta*np.matmul(delta_o, f_net_aux)

        # Reajustando os pesos
        for i in range(len(hidden_units)-1, -1, -1):
            delta = delta_h[i][:, np.newaxis]
            f_net_aux = np.concatenate((f_net_h[i],np.ones(1)))[np.newaxis, :]            
            if i == 0:
                x_aux = np.concatenate((x,np.ones(1)))[np.newaxis, :]
                hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, x_aux) + momentum_h[i]
                momentum_h[i] = -(-2)*eta*np.matmul(delta, x_aux)
            else:
                f_net_aux = np.concatenate((f_net_h[i-1],np.ones(1)))[np.newaxis, :]
                hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, f_net_aux) + momentum_h[i]
                momentum_h[i] = -(-2)*eta*np.matmul(delta, f_net_aux)

        error = sum(error*error)

        return hidden_weights, output_weights, error, momentum_h, momentum_o

    def fit(self, dataset, train_size, verbose=False):
        hidden_units = self.hidden_units
        n_classes = self.n_classes
        learning_rate = self.learning_rate
        delta_error = self.delta_error
        
        # Train-Test Split
        train, test = self.train_test_split(dataset, train_size)

        # Inicializando as camadas
        hidden_layers = len(hidden_units)
        momentum_o = 0
        momentum_h = []
        hidden_weights = []

        for i in range(hidden_layers):
            if i==0:
                aux = np.zeros((hidden_units[i], dataset.X.shape[1] + 1))
            else:
                aux = np.zeros((hidden_units[i], hidden_units[i-1] + 1))
            hidden_weights.append(aux)
            momentum_h.append(aux)

        # Preenchendo as camadas escondidas com uma distribuição normal
        for i in range(hidden_layers):
            for j in range(hidden_units[i]):
                if i==0:
                    for k in range(dataset.X.shape[1] + 1):
                        hidden_weights[i][j][k] = random.uniform(-1, 1)
                else:
                    for k in range(hidden_units[i-1]+1):
                        hidden_weights[i][j][k] = random.uniform(-1, 1)

        # Preenchendo a camada de saída com uma distribuição normal
        output_weights = np.zeros((n_classes, hidden_units[len(hidden_units)-1]+1))
        for i in range(n_classes):
            for j in range(hidden_units[hidden_layers-1]+1):
                output_weights[i][j] = random.uniform(-1, 1)

        # Épocas
        if verbose:
            print('Epoch | Erro')
        epoch = 0
        errors_list = [1,0]
        delta = 10
        while(abs(delta) > delta_error):
            sum_errors = 0
            for i in range(1,train.X.shape[0]):
                # Forward
                f_net_o, f_net_h = self.forward(train.X[i,:], hidden_weights, output_weights)      
                # Backward
                hidden_weights,output_weights,error,momentum_h,momentum_o = self.backward(train,i,hidden_weights,output_weights,f_net_o,f_net_h,learning_rate,hidden_units,momentum_h,momentum_o,n_classes)
                sum_errors += error
            errors_list.append(sum_errors)
            delta = errors_list[-1] - errors_list[-2]
            if verbose:
                print(' ', epoch, '  |', sum_errors)
            epoch += 1
        
        if not verbose:
            print('Last epoch: {} | Error: {}'.format(epoch, sum_errors))
            
        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.train = train
        self.test = test



    def predict(self, X):
        train = self.train
        test = self.test
        hidden_weights = self.hidden_weights
        output_weights = self.output_weights
        y_hat, _ = self.forward(X, hidden_weights, output_weights)
        return np.argmax(y_hat)
    
    def score(self):
        train = self.train
        test = self.test
        counter = 0
        for i in range(test.X.shape[0]):
            y_hat = self.predict(test.X[i,:])
            y = np.argmax(test.Y[i,:])
            if y == y_hat:
                counter += 1
        return counter/test.X.shape[0]
 