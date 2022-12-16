# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:25:44 2022

@author: juan_
"""
# Función para calcular la matriz de distancias

import random
import numpy as np
import matplotlib.pyplot as plt



def distancias(localizaciones): 
    # localizaciones = vector con las coordenadas de cada localización
    n = localizaciones.shape[0]   # calcula el número de localizaciones (n)
    d = np.zeros([n, n])    # crea matriz n x n
    for i in range(n):
        for j in range(n):
            d[i][j] = np.sqrt(np.square(localizaciones[i][0] - localizaciones[j][0]) + np.square(localizaciones[i][1] - localizaciones[j][1]))
    
    return d            # d será simétrica y con la diagonal = 0

def krn(i,j):   # Función Kronecker-Delta
    if i==j:
        return 1.0
    else:
        return 0.0
 




# Función de activación de cada nodo en la red
def f(u):
     return 0.5*(1+np.tanh(alfa*u))
 
    
def entrenar(A,B,C,D):
    
    # actualizamos pesos
    w = np.zeros([n**2, n**2])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):                    
                    w1 = -A*krn(i, k)*(1-krn(j, l))
                    w2 = -B*krn(j,l)*(1-krn(i, k))
                    w3 = -C                         
                    w4 = -D*d[i][k]*(1-krn(i, k))*(krn(j+1,l) + krn(j-1,l))    
                    w[X[(i, j)]][X[(k, l)]] = w1 + w2 + w3 + w4
    return w

def update(u, C):
    # update is done asynchronously
    # to make update synchronous replace C*(n+1) with a bias vector containing C*(n+sigma)
   
    for iteracion in range(5*n**2):
        i = random.randint(0, n-1)
        x = random.randint(0, n-1)
        u[X[(i, x)]][0] = f(np.dot(u.transpose(), w[:, X[(i, x)]]) + C*(n+1))
    return u


def predecir(n,totalNodos,A,B,C,D,iteraciones):
    
    u = np.zeros([totalNodos, 1])
    # Inicializar con valores aleatorios
    for i in range(n):
        for j in range(n):
            u[i][0] = random.uniform(0,0.03)
    
    prev_error = E(u, A, B, C, D)
    repeated = 0
    max_repeat = 15
    for iteracion in range(iteraciones):
        u = update(u, C)
        error = E(u, A, B, C, D)
        
        if error == prev_error:
            repeated += 1
        else:
            repeated = 0

        if repeated > max_repeat:
            break
        prev_error = error
    ret = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            ret[i][j] = u[X[(i, j)]][0]
    return ret
    
def E(u, A, B, C, D):
    E1 = 0
    
    for j in range(n):
        for i in range(n):
            for k in range(n):
                if k != i:
                    E1 += u[X[(j, i)]][0] * u[X[(j, k)]][0]
    E1 *= (A/2.0)

    E2 = 0
    for j in range(n):
        for l in range(n):
            for i in range(n):
                if l != j:
                    E2 += u[X[(i,j)]][0] * u[X[(i,l)]][0]
    E2 *= (B/2.0)

    E3 = 0
    for i in range(n):
        for j in range(n):
            E3 += u[X[(i, j)]][0]
    E3 = (C/2.0)*(E3 - n)**2    

    E4 = 0
    for i in range(n):
        for k in range(n):
            for j in range(n):
                if (k != i):
                    if 0 < i < n-1:
                        E4 += d[i][k]*u[X[(i, j)]][0]*(u[X[(k, i+1)]][0] + u[X[(k, i-1)]][0])
                    elif i == n-1:
                        E4 += d[i][k]*u[X[(i, j)]][0]*(u[X[(k, i-1)]][0] + u[X[(k, 0)]][0])
                    elif i == 0:
                        E4 += d[i][k]*u[X[(i, j)]][0]*(u[X[(k, i+1)]][0] + u[X[(k, 0)]][0])
                   
                   
                    
               
    E4 *= (D/2.0)

    return E1 + E2 + E3 + E4   





    

if __name__ == '__main__':
   
    random.seed(5) 
    n = 10  # numero de localizaciones
    # Crear vector de localización de las ciudades
    localizaciones = np.zeros([n, 2])
    localizaciones[0] = (0.25, 0.16)
    localizaciones[1] = (0.85, 0.35)
    localizaciones[2] = (0.65, 0.24)
    localizaciones[3] = (0.70, 0.50)
    localizaciones[4] = (0.15, 0.22)
    localizaciones[5] = (0.25, 0.78)
    localizaciones[6] = (0.40, 0.45)
    localizaciones[7] = (0.90, 0.65)
    localizaciones[8] = (0.55, 0.90)
    localizaciones[9] = (0.60, 0.25)
    
    # Calcula matriz distancias, dij : distancia entre i y j
    d = distancias(localizaciones)
    
    totalNodos = n**2  
    global alfa
    alfa = 50.0
    # matriz de pesos de la red neuronal
    w = np.zeros([totalNodos, totalNodos])   # creo que no es así , deberia ser w nxn  
    
    # listados para realizar las conversiones
    X = {}
        
    indice = 0
    for i in range(n):
        for j in range(n):           
            X[(i, j)] = indice
            indice += 1
    
   
    iteraciones = 2000
    summation = 0
    mini = 1000
    maxi = -1
   
    horizontal = np.arange(iteraciones)
    minimos = np.zeros(iteraciones)
    valores = np.zeros(iteraciones)
    maximos = np.zeros(iteraciones)
    medias = np.zeros(iteraciones)
    solucion = np.zeros([n, n]) 

    for iteracion in range(iteraciones):
        x = []
        y = []
        print("Iteración:", iteracion+1)
        w = entrenar(A=100.0, B=100.0, C=90.0, D=110.0)
        
        v = predecir(n,totalNodos,A=100.0, B=100.0, C=90.0, D=110.0,iteraciones=iteraciones)
          
        
        dist = 0
        prev_row = -1
        
        for col in range(v.shape[1]):
            for row in range(v.shape[0]):
                
                if v[row][col] == 1:
                    if prev_row != -1:
                        
                        x.append(localizaciones[prev_row][0])
                        y.append(localizaciones[prev_row][1])
                       
                        dist += d[prev_row][row]
                        #print("From City {} To City {}".format(prev_row + 1, row + 1))
                    prev_row = row
                    break
            if (col ==v.shape[1]-1):
                x.append(localizaciones[row][0])
                y.append(localizaciones[row][1])
                    
       
        summation += dist
        
       
        
        mini = min(mini, dist)
        maxi = max(maxi, dist)
        
        medias[iteracion] =summation*10/(iteracion+1)        
        minimos[iteracion] =mini*10
        maximos[iteracion] =maxi*10
        
        if(dist == mini):
            minimos[iteracion] =dist*10
            solucion = v
        if(dist == maxi):
            maximos[iteracion] =dist*10
        valores[iteracion] =dist*10
            
            
        plt.subplot(2, 1, 1)
        plt.plot(x, y,marker ='o')
        #plt.plot(x, y, 'r*')
        
        
        plt.subplot(2, 1,2)
        
        
        plt.plot(horizontal,valores,'k.',markersize=1)
        plt.plot(horizontal,maximos,'r.',markersize=1)
        plt.plot(horizontal,medias,'y.',markersize=1)
        plt.plot(horizontal,minimos,'g.',markersize=1)
        
        plt.show()
        
        
        print("Distance:", dist*10, " , Min: ", mini*10, ", Media: ", summation*10/(iteracion+1) , "\n")
        
    print("\nMin: {}\nMax: {}\nAverage: {}".format(mini*10, maxi*10, summation*10 / iteraciones))
    x = []
    y = []
    prev_row = -1
    for col in range(solucion.shape[1]):
        for row in range(solucion.shape[0]):
            
            if solucion[row][col] == 1:
                if prev_row != -1:
                    
                    x.append(localizaciones[prev_row][0])
                    y.append(localizaciones[prev_row][1])
                   
                    dist += d[prev_row][row]
                    print("From City {} To City {}".format(prev_row + 1, row + 1))
                prev_row = row
                break
        if (col ==solucion.shape[1]-1):
            x.append(localizaciones[row][0])
            y.append(localizaciones[row][1])
            
    plt.plot(x, y,marker ='o')
    plt.show()
    
   
   
    
    