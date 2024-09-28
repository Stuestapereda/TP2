import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import heapq #para encontrar los n elementos más pequeños de una variable set

df = pd.read_csv("reviews_sentiment.csv", sep=';')

#Promedio de palabras de comentarios valorados con una estrella
df_1 = df[df["Star Rating"]==1]
word_count = df_1["wordcount"]
promedio = np.mean(word_count)
print("Promedio de palabras de comentarios valorados con una estrella: ", promedio)

#Eliminar las filas con columnas NA
df_filtrado = df.dropna() #Solo son 26 filas con NA

#0 = negativo, 1 = positivo
def str_to_num(ts):
    return 0 if ts =="negative" else 1

df_filtrado['titleSentiment'] = df_filtrado['titleSentiment'].apply(str_to_num)
#Dividir el dataset en entrenamiento y test
entrenamiento, prueba = train_test_split(df_filtrado, test_size=0.3, random_state=42)
"""
Variable objetivo: Star Rating
Variables predictoras: wordcount, title sentiment, sentiment value
"""
def distancia_euclidiana(datos1,datos2):
    datos1 = np.array(datos1)
    datos2 = np.array(datos2)
    
    diferencia = datos1 - datos2
    return np.sqrt(np.sum(diferencia**2, axis=1))

def knn_model(data,variable_clases,variables_predictoras,k,nuevo_dato):
    dic_clases={}
    clases = data[variable_clases].unique()
    for clase in clases:
        data_por_clase = data[data[variable_clases]==clase] #Filtramos los datos por clase
        data_por_variable = [] #Inicializamos un array para almacenar los datos de cada atributo/variable
        for variable in variables_predictoras:
            data_por_variable.append(data_por_clase[variable]) #Almacenamos los datos de cada atributo/variable en un array
        dic_clases[clase] = np.array(data_por_variable).T #Almacenamos los datos de cada clase en un diccionario
    
    #for clase in dic_clases:
    #    print(f"clase {clase}: {dic_clases[clase]}")

    #Preprocesado el nuevo dato
    nuevo_dato_pre_procesado = []
    for variable in variables_predictoras:
        nuevo_dato_pre_procesado.append(float(nuevo_dato[variable]))

    #Calculo de las distancias
    dic_distancias_por_clase = {}
    for clase in dic_clases:
        datos_por_clase_2 = dic_clases[clase]

        distancia = distancia_euclidiana([nuevo_dato_pre_procesado],datos_por_clase_2)
        dic_distancias_por_clase[clase] = distancia

    #for clase in dic_distancias_por_clase:
    #    print(f"Distancia de la clase {clase}: {dic_distancias_por_clase[clase]}")

    distancias = []
    for clase in dic_distancias_por_clase:
        # Convertir el array de NumPy a una tupla antes de agregarlo al set
        distancias.extend(list(dic_distancias_por_clase[clase]))

    distancias_unicas = list(set(distancias)) #eliminando las distancias repetidas
    
    k_distancias = heapq.nsmallest(k,distancias_unicas)

    #pre incializando la variable dic_clases_mas_cercanas
    dic_clases_mas_cercanas = {}
    for clase in clases:
        dic_clases_mas_cercanas[clase] = 0

    for d in k_distancias:
        for clase in dic_distancias_por_clase:
            if d in dic_distancias_por_clase[clase]:
                dic_clases_mas_cercanas[clase] += 1
    
    clase = max(dic_clases_mas_cercanas, key=dic_clases_mas_cercanas.get)
    
    return clase

predicciones = []
for i in range(len(prueba)):
    clase_predicha=knn_model(entrenamiento,"Star Rating",["wordcount","titleSentiment","sentimentValue"],3,prueba.iloc[i])
    predicciones.append(clase_predicha)

prueba["prediccion"] = predicciones

print("La clase predicha es: ", prueba["prediccion"])
print("La clase real es: ", prueba["Star Rating"])
