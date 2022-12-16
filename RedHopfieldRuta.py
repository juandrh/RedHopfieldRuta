# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:25:44 2022

@author: juan_
"""
# Función para calcular la matriz de distancias

import random



def distancias(ciudades): 
    # ciudades = vector con las coordenadas de cada ciudad
	n = ciudades.shape[0]   # calcula el número de ciudades (n)
	d = np.zeros([n, n])    # crea matriz n x n
	for i in range(n):
		for j in range(n):
			d[i][j] = np.sqrt(np.square(ciudades[i][0] - ciudades[j][0]) + np.square(ciudades[i][1] - ciudades[j][1]))
	return d            # d será simétrica y con la diagonal = 0

def krn(i,j):   # Función Kronecker-Delta
    if i==j:
        return 1.0
    else:
        return 0.0
 

def denergia(linea):
    global n_entradas, pesos, nodos_entrada, sum_lin_pesos
    temp=0.0
    for i in range(n_entradas):
        temp=temp+(pesos[linea][i])*(nodos_entrada[i])
    return 2.0*temp-sum_lin_pesos[linea]


# Función de activación de cada nodo en la red
 def v(alfa, u):
     return 0.5*(1+np.tanh(alfa*u))
 
    
def entrenar(n,A,B,C,D,d):
    global n_entradas, pesos, nodos_entrada, sum_lin_pesos
    # actualizamos pesos
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    w1 = -A*krn(i, k) (1-krn(j, l))
                    w2 = -B*krn(j,l)(1-krn(i, k))
                    w3 = -C                         
                    w4 = -D*d[i][k]*(1-krn(i, k))*(krn(l,j+1) + krn(l, j-1))    
                    w[X[(i, j)]][X[(k, l)]] = w1 + w2 + w3 + w4

def predecir(n,totalNodos,A,B,C,D,iteraciones):
    
    u = np.zeros([totalNodos, 1])
    # Inicializar con valores aleatorios
    for i in range(n):
        for j in range(n):
            u[i][0] = random.random()
    
    prev_error = E(u, A, B, C, D)
    repeated = 0
    max_repeat = 15
    for iteration in range(iteraciones):
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
    
def E(u, A, B, C, D):
    E1 = 0
    n = self.cities
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
                if i != k:
                    E4 += d[i][k]*u[X[(i, j)]][0]*(u[X[(k, i+1)]][0] + u[X[(k, i-1)]][0])
               
    E4 *= (D/2.0)

    return E1 + E2 + E3 + E4   
    

if __name__ == '__main__':
   
    random.seed(5) 
    n = 10  # numero de ciudades
    # Crear vector de localización de las ciudades
    ciudades = np.zeros([n, 2])
	ciudades[0] = (0.06, 0.70)
	ciudades[1] = (0.08, 0.90)
	ciudades[2] = (0.22, 0.67)
	ciudades[3] = (0.30, 0.20)
	ciudades[4] = (0.35, 0.95)
	ciudades[5] = (0.40, 0.15)
	ciudades[6] = (0.50, 0.75)
	ciudades[7] = (0.62, 0.70)
	ciudades[8] = (0.70, 0.80)
	ciudades[9] = (0.83, 0.20)
    
    # Calcula matriz distancias, dij : distancia entre i y j
    d = distancias(ciudades)
    
    totalNodos = n**2
    alfa = 50.0
    # matriz de pesos de la red neuronal
    w = np.zeros([totalNodos, totalNodos])    
    
    # listados para realizar las conversiones
    X = {}
        
    indice = 0
    for i in range(n):
        for j in range(n):           
            X[(i, j)] = indice
            indice += 1
    
   iteraciones = 5 * n**2
    
    