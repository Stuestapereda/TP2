import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import heapq #para encontrar los n elementos más pequeños de una variable set

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

