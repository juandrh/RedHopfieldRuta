# -*- coding: utf-8 -*-

# Función para calcular la matriz de distancias

import random
import numpy as np
import matplotlib.pyplot as plt


# Cálculo de la matriz de distancias
def distancias(localizaciones): 
    # localizaciones = vector con las coordenadas de cada localización
    n = localizaciones.shape[0]   # calcula el número de localizaciones (n)
    d = np.zeros([n, n])    # crea matriz n x n
    for i in range(n):
        for j in range(n):
            d[i][j] = np.sqrt(np.square(localizaciones[i][0] - localizaciones[j][0]) + np.square(localizaciones[i][1] - localizaciones[j][1]))
    
    return d            # d será simétrica y con la diagonal = 0

 # Función Kronecker-Delta. Aprendizaje Hebbian
def krn(i,j):  
    if i==j:
        return 1.0
    else:
        return 0.0
 

# Función de activación sigmoidal de cada nodo en la red
def f(u):    
    return 0.5*(1.0+np.tanh(alfa*u))
 
# Cálculo de la matriz de pesos por el entrenamiento   
def entrenar(A,B,C,D):    
    # actualizamos pesos   
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

# Actualización de los estados externos de la red
def actualizar(v, C): 
   
    for iteracion in range(5*n**2):
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        v[X[(i, j)]][0] = f(np.dot(v.transpose(), w[:, X[(i, j)]]) + C*(n+1.0))        
    return v



def predecir(n,totalNodos,A,B,C,D,iteraciones):    
    v = np.zeros([totalNodos, 1])
    # Inicializar con valores aleatoriosa
    for i in range(n):
        for j in range(n):
            v[n*j+i][0] = random.uniform(0, 0.03)
    
    energiaAnterior = energia(v, A, B, C, D)
    repetido = 0
    maxRepeticiones= 15   # límite para determinar si converge la función de energía
    for iteracion in range(iteraciones):
        v =actualizar(v, C)  
        en = energia(v, A, B, C, D)        
        if en == energiaAnterior:
            repetido += 1
        else:
            repetido = 0
            
        if repetido >maxRepeticiones:            
            break
        energiaAnterior = en
        if (iteracion < 1000):
            energias[iteracion] = en +energias[iteracion] 
    
    ret = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            ret[i][j] = v[X[(i, j)]][0]
   
    return ret

# Función de Energía    
def energia(v, A, B, C, D):
    E1 = 0
    
    for j in range(n):
        for i in range(n):
            for k in range(n):
                if k != i:
                    E1 += v[X[(j, i)]][0] * v[X[(j, k)]][0]
    E1 *= (A/2.0)

    E2 = 0
    for j in range(n):
        for l in range(n):
            for i in range(n):
                if l != j:
                    E2 += v[X[(i,j)]][0] * v[X[(i,l)]][0]
    E2 *= (B/2.0)

    E3 = 0
    for i in range(n):
        for j in range(n):
            E3 += v[X[(i, j)]][0]
    E3 = (C/2.0)*(E3 - n)**2    

    E4 = 0
    for i in range(n):
        for k in range(n):
            for j in range(n):
                if (k != i):
                    if 0 < i < n-1:
                        E4 += d[i][k]*v[X[(i, j)]][0]*(v[X[(k, i+1)]][0] + v[X[(k, i-1)]][0])
                    elif i == n-1:
                        E4 += d[i][k]*v[X[(i, j)]][0]*(v[X[(k, i-1)]][0] + v[X[(k, 0)]][0])
                    elif i == 0:
                        E4 += d[i][k]*v[X[(i, j)]][0]*(v[X[(k, i+1)]][0] + v[X[(k, 0)]][0])
            
    E4 *= (D/2.0) 
    return E1 + E2 + E3 + E4   
    

if __name__ == '__main__':
   
    random.seed(5)  # semilla de números aleatorios
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
    '''localizaciones[10] = (0.35, 0.26)
    localizaciones[11] = (0.75, 0.40)
    localizaciones[12] = (0.55, 0.34)
    localizaciones[13] = (0.80, 0.60)
    localizaciones[14] = (0.25, 0.12)
    localizaciones[15] = (0.35, 0.68)
    localizaciones[16] = (0.30, 0.55)
    localizaciones[17] = (0.80, 0.75)
    localizaciones[18] = (0.65, 0.80)
    localizaciones[19] = (0.50, 0.15)
    localizaciones[20] = (0.15, 0.16)
    localizaciones[21] = (0.79, 0.49)
    localizaciones[22] = (0.59, 0.30)
    localizaciones[23] = (0.85, 0.65)
    localizaciones[24] = (0.30, 0.17)
    localizaciones[25] = (0.36, 0.78)
    localizaciones[26] = (0.31, 0.57)
    localizaciones[27] = (0.99, 0.89)
    localizaciones[28] = (0.01, 0.02)
    localizaciones[29] = (0.53, 0.19) ''' 
    
    # Calcula matriz distancias, dij : distancia entre i y j
    d = distancias(localizaciones)
    
    totalNodos = n**2  
    global alfa
    alfa = 50.0
    A = 100.0 # 500.0
    B = 100.0 # 500.0
    C = 90.0 # 350.0
    D = 110.0 # 550.0
    
    # matriz de pesos de la red neuronal
    w = np.zeros([totalNodos, totalNodos])     
    
    # listado para realizar la conversión
    X = {}        
    indice = 0
    for i in range(n):
        for j in range(n):           
            X[(i, j)] = indice
            indice += 1
    
   
    iteraciones = 200
    summation = 0
    mini = 1000
    maxi = -1
   
    # vectores para almacenar resultados a mostrar
    horizontal = np.arange(iteraciones)
    minimos = np.zeros(iteraciones)
    valores = np.zeros(iteraciones)
    maximos = np.zeros(iteraciones)
    medias = np.zeros(iteraciones)
    
    solucion = np.zeros([n, n]) 
    
    w = entrenar(A, B, C, D)
    
    for iteracion in range(iteraciones):
        x = []
        y = []
        energias = np.zeros(1000)
        localizacionesRecorridas=0
        print("Iteración:", iteracion+1)           
        v = predecir(n,totalNodos,A, B, C, D,iteraciones=1000)   

                   
        dist = 0
        prev_row = -1        
        for col in range(v.shape[1]):
            for row in range(v.shape[0]):                
                if v[row][col] == 1:
                    if prev_row != -1:                        
                        x.append(localizaciones[prev_row][0])
                        y.append(localizaciones[prev_row][1])                       
                        dist += d[prev_row][row]   
                        localizacionesRecorridas = localizacionesRecorridas +1
                    prev_row = row
                    break
            if (col ==v.shape[1]-1):
                x.append(localizaciones[row][0])
                y.append(localizaciones[row][1])
        x.append(x[0])
        y.append(y[0])            
        summation += dist                  
        mini = min(mini, dist)
        maxi = max(maxi, dist)        
        medias[iteracion] =summation*10/(iteracion+1)        
        minimos[iteracion] =mini*10
        maximos[iteracion] =maxi*10
        # Añadir a mínimos si ha recorrido todos los puntos
        if(dist == mini and localizacionesRecorridas == n-1):
            minimos[iteracion] =dist*10
            solucion = v
        if(dist == maxi):
            maximos[iteracion] =dist*10
        valores[iteracion] =dist*10
            


        # Gráficos de soluciones parciales             
        plt.subplot(3, 1, 1)
        plt.plot(x, y,marker ='o')
        
        plt.subplot(3, 1,2)    
        plt.plot(horizontal,valores,'k.',markersize=1)
        plt.plot(horizontal,maximos,'r.',markersize=1)
        plt.plot(horizontal,medias,'y.',markersize=1)
        plt.plot(horizontal,minimos,'g.',markersize=1)  
        
        plt.subplot(3, 1,3)
        h = np.arange(1000)
                       
        plt.plot(h,energias,'k.',markersize=1)
        plt.show()
        print("Total localizaciones recorridas: ", localizacionesRecorridas+1)  
        if(localizacionesRecorridas<n-1):
            print("Solución incompleta.")
        print("Distancia:", dist*10, " , Min: ", mini*10, ", Media: ", summation*10/(iteracion+1) , "\n")
        
    print("\nMin: {}\nMax: {}\nMedia: {}".format(mini*10, maxi*10, summation*10 / iteraciones))
    x = []
    y = []
    prev_row = -1
    
    # Gráfico de solución final
    for col in range(solucion.shape[1]):
        for row in range(solucion.shape[0]):
            
            if solucion[row][col] == 1:
                if prev_row != -1:
                    
                    x.append(localizaciones[prev_row][0])
                    y.append(localizaciones[prev_row][1])
                   
                    dist += d[prev_row][row]
                    print("Del punto {} al punto {}".format(prev_row + 1, row + 1))
                prev_row = row
                break
        if (col ==solucion.shape[1]-1):
            x.append(localizaciones[row][0])
            y.append(localizaciones[row][1])
    x.append(x[0])
    y.append(y[0])          
    plt.plot(x, y,marker ='o')
    plt.show()
    
   
   
    
    