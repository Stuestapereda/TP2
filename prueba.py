import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import heapq #para encontrar los n elementos más pequeños de una variable set

"""
#Distancia euclidiana
def distancia_euclidiana(datos1,datos2):
    datos1 = np.array(datos1)
    datos2 = np.array(datos2)

    diferencia = datos1 - datos2
    return np.sqrt(np.sum(diferencia**2, axis=1))

datos1=[1,2,3]
datos2=[4,5,6]

#print(distancia_euclidiana(datos1,datos2))

datos3=[1,2]
datos4=[4,5]

#print(distancia_euclidiana(datos3,datos4))

datos5=np.array([[1,2,3],[4,5,6]])
datos6=np.array([[4,5,6]])

print(distancia_euclidiana(datos6,datos5))
"""

distancias_por_clase = {"clase1": [1,2,3], "clase2": [2,5,6]}
k_distancias = [1,2]

conteo_clases = {clase: sum(d in distancias_por_clase[clase] for d in k_distancias) for clase in distancias_por_clase}
print(conteo_clases)

conteo_clases = {}
for clase in distancias_por_clase:
    conteo_clases[clase] = 0
    for d in k_distancias:
        if d in distancias_por_clase[clase]:
            peso = 1/(d**2)
            conteo_clases[clase] += 1*peso

print(conteo_clases)

k=2
vecinos=0
conteo_clases = {}
for d in k_distancias:
    for clase in distancias_por_clase:
        if clase not in conteo_clases:
            conteo_clases[clase] = 0
        if d in distancias_por_clase[clase]:
            vecinos += 1
            if vecinos <= k:
                conteo_clases[clase] += 1

print(conteo_clases)  