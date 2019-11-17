import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def cria_vetor_t(df):
    test_dataset=pd.DataFrame(columns=["IDADE (anos)","Peso (kg)","Carga Final:","VO2 medido máximo (mL/kg/min)"])
    t_vector=np.ones(1172)
    rand=np.random.randint(0,1171, 172)
    for e in rand:
        t_vector[e]=0
        test_dataset=test_dataset.append(df.loc[e])

    df.drop(df.index[rand])
    return t_vector, test_dataset, df

def altera_target(Y):
    target=Y.to_numpy()
    for i, label in enumerate(target):
        if label < 40:
            target[i]=0
        elif label > 39 and label < 60:
            target[i]=1
        else:
            target[i]=2
    return target



if __name__ == '__main__':
    #leitura e separacao dos dados em conjunto de treino, e conjunto de teste

    df = pd.read_csv('dados.csv', delimiter = '\s+', index_col=False)
    t_vector, test_df, train_df=cria_vetor_t(df)

    print(t_vector)

    #neste modelo a nossa variavel a ser descoberta sera a idade dos pacientes, e elas serao divididas da seguinte forma:
    #0: idade entre 18 e 39, 1: idade entre 40 e 59, 2: idade 60 para cima.

    Y = train_df.iloc[:, 0]
    X = train_df.iloc[:, 1:4]

    Y=altera_target(Y)

    # agora que os dados ja estao prontos para serem utilizados, comecaremos nosso problema

    gmm = GaussianMixture(n_components = 3)
    gmm = gmm.fit(X, Y)

    print(gmm.means_)
