
'''aluno: Rennan de Lucena Gaio'''

import pandas as pd
import numpy as np

'''classe da regressao linear para aprendizado'''

class LinearRegression:
    """
    Implementação da regressao linear
    """
    #função de inicialização da nossa classe. Aqui serão guardados tanto os pesos de nosso perceptron na variável w, como o learning rate na variável LR

    def __init__(self):
        self.w = np.zeros(3)

    #função que classifica o conjunto de pontos de entrada, em que data e um vetor com features-dimenções e N exemplos
    #para fazer a classificacao dos pontos basta aplicar o produto interno entre os pesos e as features do seu ponto, caso seja maior que 0, entao elas pertencem a uma classe (1), caso seja
    #menor, elas pertencem a outra classe (-1)
    def predict(self, data):
        predicted_labels=[]
        for point in data:
            predicted_labels.append(np.inner(self.w, point))

        return predicted_labels


    #função de aprendizado e atualização dos pesos do algoritmo, aqui está implementado o método de minimos quadrados para n dimenções
    #o operador @ é um operador de multiplicação de matrizes. ele é específico do python 3 com o numpy, mas ele foi muito util nessa situação.
    def fit(self, X, y):
        A=(X.T)@X
        B=(X.T)@y
        #essa função tem por objetivo resolver sistemas do tipo Ax=B, em que você passa como parâmetros o A e o B e ele te retorna o valor de x.
        #essa forma é muito mais eficiente de fazer do que calcular a inversa da função e depois fazer outra multiplicação de matriz.
        self.w= np.linalg.solve(A, B)
        return self.w


def exercise_1_a(dataset):

    print("Questão 1 A")
    X = []
    Y = dataset['y']

    for element in dataset['x']:
        X.append([1,element])

    reg = LinearRegression().fit(np.array(X), Y)

    print("Modelo M1: ")
    print(reg)

    X=[]
    for element in dataset['x']:
        X.append([1,element, element**2, element**3, element**4])

    reg = LinearRegression().fit(np.array(X), Y)

    print("Modelo M2: ")
    print(reg)

def exercise_1_b(dataset):

    print("Questão 1 B")
    X = []
    Y = dataset['y']

    for element in dataset['x']:
        X.append([1,element])

    reg = LinearRegression()
    pesos = reg.fit(np.array(X), Y)

    erro_quadratico1 = 0

    y_estimado=reg.predict(np.array(X))

    for i in range(len(y_estimado)):
        erro_quadratico1+= (Y[i]-y_estimado[i])**2

    print("Erro modelo M1: ")
    print(erro_quadratico1/6)

    X=[]
    for element in dataset['x']:
        X.append([1,element, element**2, element**3, element**4])

    reg = LinearRegression()
    pesos = reg.fit(np.array(X), Y)

    erro_quadratico2 = 0

    y_estimado=reg.predict(np.array(X))

    for i in range(len(y_estimado)):
        erro_quadratico2+= (Y[i]-y_estimado[i])**2

    print("Erro modelo M2: ")
    print(erro_quadratico2/6)

    print("razão M1/M2: ")
    print(erro_quadratico1/erro_quadratico2)

def exercise_1_c(dataset):

    print("Questão 1 C")
    X = []
    Y = dataset['y']
    new_X = []

    dados=[848, 912]

    for element in dataset['x']:
        X.append([1,element])

    for element in dados:
        new_X.append([1,element])

    reg = LinearRegression()
    pesos=reg.fit(np.array(X), Y)

    y_estimado=reg.predict(np.array(new_X))

    print("valores estimados modelo M1: ")
    print(y_estimado)


    X=[]
    new_X = []
    for element in dataset['x']:
        X.append([1,element, element**2, element**3, element**4])

    for element in dados:
        new_X.append([1,element, element**2, element**3, element**4])


    reg = LinearRegression()
    pesos=reg.fit(np.array(X), Y)

    y_estimado=reg.predict(np.array(new_X))

    print("valores estimados modelo M2: ")
    print(y_estimado)


def exercise_1_d(dataset):

    print("Questão 1 D")
    X = []
    Y = dataset['y']
    new_X = []
    y_true = [155900, 156000]

    dados=[848, 912]

    for element in dataset['x']:
        X.append([1,element])

    for element in dados:
        new_X.append([1,element])

    reg = LinearRegression()
    pesos=reg.fit(np.array(X), Y)

    erro_quadratico1 = 0

    y_estimado=reg.predict(np.array(new_X))

    for i in range(len(y_estimado)):
        erro_quadratico1+= (y_true[i]-y_estimado[i])**2

    print("Erro modelo M1: ")
    print(erro_quadratico1/6)


    X=[]
    new_X = []
    for element in dataset['x']:
        X.append([1,element, element**2, element**3, element**4])

    for element in dados:
        new_X.append([1,element, element**2, element**3, element**4])


    reg = LinearRegression()
    pesos=reg.fit(np.array(X), Y)

    erro_quadratico2 = 0

    y_estimado=reg.predict(np.array(new_X))

    for i in range(len(y_estimado)):
        erro_quadratico2+= (y_true[i]-y_estimado[i])**2

    print("Erro modelo M2: ")
    print(erro_quadratico2/6)

    print("razão M1/M2: ")
    print(erro_quadratico1/erro_quadratico2)




if __name__ == '__main__':

    dataset = pd.read_csv('dataset1.csv', sep=',')
    exercise_1_a(dataset)
    print("#####################################")
    exercise_1_b(dataset)
    print("#####################################")
    exercise_1_c(dataset)
    print("#####################################")
    exercise_1_d(dataset)
