import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Leitura do csv
df = pd.read_csv('semeion.data', sep=' ', lineterminator='\n')
print('Shape do arquivo:', df.shape)

# Preparação do dataset
digitos = prepara_dataset(df, 10)

# Criando a rede
mlp = MLP(hidden_units=[30], n_classes=10, learning_rate=0.5, delta_error=1e-2)
print('\nMLP criada ({})'.format(mlp))

# Treinando a rede
print('\nTreinando:\n')
mlp.fit(digitos, train_size=0.7, verbose=True)

# Score
print('\nAcurácia:', mlp.score())

# Resultado
plt.figure(figsize=(17,5))
for i in range(1,17):
    plt.subplot(2,8,i)
    rand = np.random.randint(0, len(df))
    real = np.argmax(digitos.Y[rand])
    pred = mlp.predict(digitos.X[rand])
    plt.imshow(digitos.X[rand].reshape(16,16), cmap='binary')
    plt.xlabel('Real: {}\nPrevisto: {}'.format(real, pred))
    plt.xticks([])
    plt.yticks([])
plt.show()