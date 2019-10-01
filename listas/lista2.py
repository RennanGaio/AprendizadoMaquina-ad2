import random as r
import numpy as np
import matplotlib.pyplot as plt

def ex1():
    joao=10000
    maria=10000

    for i in range(5000):
        if (r.randint(0,1) > 0.5):
            maria+= 100
            joao-= 100
        else:
            joao+= 100
            maria-= 100

        if not(joao) or not(maria):
            break

    print ("quantidade do joao ", joao)
    print ("quantidade da maria ", maria)

def ex2(N=5000):
    jogadores=[10000]*500

    for rodada in range(N):
        for i in range(len(jogadores)):
            if jogadores[i]:
                for j in range(len(jogadores)):
                    if jogadores[j]:
                        if i != j:
                            if (r.randint(0,1) > 0.5):
                                jogadores[i]+= 100
                                jogadores[j]-= 100
                            else:
                                jogadores[j]+= 100
                                jogadores[i]-= 100

    maximo=max(jogadores)
    jogadores=np.array(jogadores)/maximo

    countpoor=0
    countrich=0
    for jogador in jogadores:
        if jogador<=0:
            countpoor+=1
        if jogador>0.9:
            countrich+=1

    print("percentagem de pobres eh ", countpoor/len(jogadores))
    print("percentagem de ricos eh ", countrich/len(jogadores))

    #np.histogram(jogadores)
    n, bins, patches = plt.hist(x=jogadores, bins=10, color='#0504aa', rwidth=0.85)
    # plt.grid(axis='rmax')
    plt.xlabel('rmax')
    plt.ylabel('jogadores')
    plt.title('My Very Own Histogram')

    plt.show()

if __name__ == '__main__':
    ex1()
    ex2(5000)
    ex2(10000)
