import random
import pandas as pd
import numpy as np
from collections import namedtuple

from MLP import MLP


def prepara_dataset(data, y_collumns, sep=','):
    y = data.iloc[:,len(data.columns)-y_collumns: len(data.columns)]
    y = np.array(y)
    X = data.iloc[:,0:len(data.columns)-y_collumns]
    X = np.array(X)
    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]
    dataset = namedtuple('datset', 'X Y')
    return dataset(X=X_values, Y=y_values)


def train_test_split(dataset, train_size):
    lenght = dataset.X.shape[0]
    x_train = dataset.X[0:int(train_size*lenght), :]
    y_train = dataset.Y[0:int(train_size*lenght), :]
    x_test = dataset.X[int(train_size*lenght):, :]
    y_test = dataset.Y[int(train_size*lenght):, :]
    dataset = namedtuple('datset', 'X Y')
    train = dataset(X=x_train, Y=y_train)
    test = dataset(X=x_test, Y=y_test)
    return train, test


# Cross Validation
def k_fold(dataset, k, verbose=False, rts='all'):
    if verbose: print('[k_fold]')

    accs = []

    # Iterando a partir do 1 até k-1
    for i in range(1, k):

        # Determinando tamanho dos folds
        foldsize = (1/k)*i
        if verbose:
            print('\nk: {}           | Train Size: {:.2f}%'.format(i, foldsize))

        # Separando em conjuntos de treinamento e teste
        train, test = train_test_split(dataset, train_size=foldsize)

        # Criando modelo
        mlp = MLP(hidden_units=[30], n_classes=10, learning_rate=0.5, delta_error=1e-2)
        
        # Treinando o modelo criado
        mlp.fit(dataset, train, dataset.X.shape[1], verbose=False)
        
        # Realizando os testes e salvando em um vetor
        accs.append(mlp.score(test))


    if rts == 'all':
        return accs
    elif rts == 'mean':
        return sum(accs)/(k-1)