import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import heapq

df = pd.read_csv("reviews_sentiment.csv", sep=';')

# Promedio de palabras de comentarios valorados con una estrella
promedio = df[df["Star Rating"]==1]["wordcount"].mean()
print("Promedio de palabras de comentarios valorados con una estrella: ", promedio)

# Eliminar las filas con columnas NA
df_filtrado = df.dropna()

# 0 = negativo, 1 = positivo
df_filtrado['titleSentiment'] = df_filtrado['titleSentiment'].map({'negative': 0, 'positive': 1})

# Dividir el dataset en entrenamiento y test
entrenamiento, prueba = train_test_split(df_filtrado, test_size=0.3, random_state=42)

def distancia_euclidiana(datos1, datos2):
    return np.linalg.norm(np.array(datos1) - np.array(datos2), axis=1)

def knn_model(data, variable_clases, variables_predictoras, k, nuevo_dato):
    dic_distancias_por_clase = {}
    nuevo_dato_pre_procesado = [float(nuevo_dato[var]) for var in variables_predictoras]

    for clase in data[variable_clases].unique():
        datos_clase = data[data[variable_clases] == clase][variables_predictoras].values
        dic_distancias_por_clase[clase] = distancia_euclidiana(nuevo_dato_pre_procesado, datos_clase)

    distancias = np.concatenate(list(dic_distancias_por_clase.values()))
    k_distancias = heapq.nsmallest(k, set(distancias))

    conteo_clases = {clase: sum(d in dic_distancias_por_clase[clase] for d in k_distancias) for clase in dic_distancias_por_clase}
    return max(conteo_clases, key=conteo_clases.get)

prueba['prediccion'] = [knn_model(entrenamiento, "Star Rating", ["wordcount", "titleSentiment", "sentimentValue"], 3, prueba.iloc[i]) for i in range(len(prueba))]

print("La clase predicha es: ", prueba["prediccion"])
print("La clase real es: ", prueba["Star Rating"])
