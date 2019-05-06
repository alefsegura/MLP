import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from MLP import MLP
from handler import prepara_dataset, train_test_split, k_fold



# Leitura do csv
df = pd.read_csv('semeion.data', sep=' ', lineterminator='\n')
print('Shape do arquivo:', df.shape)


# Preparação do dataset
digitos = prepara_dataset(df, 10)

'''
# Criando a rede
mlp = MLP(hidden_units=[30], n_classes=10, learning_rate=0.5, delta_error=1e-2)
print('\nMLP criada ({})'.format(mlp))

# TDividindo em conjunto de teste e treinamentos
train, test = train_test_split(digitos, 0.7)

# Treinando a rede
print('\nTreinando:\n')
mlp.fit(digitos, train, digitos.X.shape[1], verbose=True)

# Score
print('\nAcurácia:', mlp.score(test))


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

'''
# Cross Validation
k = 5
acuracias = k_fold(digitos, k, verbose=True)
print('\n\nCross Validation mean accuracy: {:.4f}%'.format((sum(acuracias)/(k-1))*100))