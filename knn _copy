import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import heapq #para encontrar los n elementos más pequeños de una variable set
from sklearn.metrics import precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

def knn_model(data,variable_clases,variables_predictoras,k,nuevo_dato,ponderacion):
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

    """
    #algoritmo con sesgo hacia la primera clase q se cuenta
    vecinos=0
    for d in k_distancias:
        for clase in dic_distancias_por_clase:
            if d in dic_distancias_por_clase[clase]:
                if vecinos <= k:
                    dic_clases_mas_cercanas[clase] += 1
                    vecinos += 1

    """
    """
    #algoritmo con sesgo: cuenta k distancias más cercanas, no k vecinos mas cercanos
    conteo_clases = {clase: sum(d in distancias_por_clase[clase] for d in k_distancias) for clase in distancias_por_clase}
    """ 
    #algoritmo con sesgo aleatorio, se consideran todos los vecinos con la misma distancia y luego se escoge aleatoriamente
    vecinos=0
    for d in k_distancias:
        clase_de_vecinos_por_distancia = []
        for clase in dic_distancias_por_clase:
            if d in dic_distancias_por_clase[clase]:
                clase_de_vecinos_por_distancia.append(clase)

        vecinos += len(clase_de_vecinos_por_distancia)
        if vecinos > k:
            clases_elegidas = np.random.choice(clase_de_vecinos_por_distancia,size=vecinos-k,replace=False)
        else:
            clases_elegidas = clase_de_vecinos_por_distancia

        if ponderacion == False:
            for c in clases_elegidas:
                dic_clases_mas_cercanas[c] += 1
        if ponderacion == True:
            for c in clases_elegidas:
                dic_clases_mas_cercanas[c] += 1/(d**2)

        if vecinos > k:
            break
    
    # Encontrar el valor máximo de conteos de vecinos
    max_valor = max(dic_clases_mas_cercanas.values())

    # Filtrar las clases que tienen el valor máximo
    clases_empate = [clase for clase, conteo in dic_clases_mas_cercanas.items() if conteo == max_valor]

    # Si hay más de una clase con el mismo número de vecinos, seleccionamos una aleatoriamente
    if len(clases_empate) > 1:
        clase = np.random.choice(clases_empate)
    else:
        clase = clases_empate[0]  # Si solo hay una clase con el valor máximo

    
    return clase

# Predicciones ponderadas
predicciones_ponderadas = [knn_model(entrenamiento, "Star Rating", ["wordcount", "titleSentiment", "sentimentValue"], 5, prueba.iloc[i], ponderacion=True) for i in range(len(prueba))]
prueba["prediccion_ponderada"] = predicciones_ponderadas

# Predicciones no ponderadas
predicciones_no_ponderadas = [knn_model(entrenamiento, "Star Rating", ["wordcount", "titleSentiment", "sentimentValue"], 5, prueba.iloc[i], ponderacion=False) for i in range(len(prueba))]
prueba["prediccion_no_ponderada"] = predicciones_no_ponderadas

# Generar las matrices de confusión
matriz_confusion_ponderada = confusion_matrix(prueba["Star Rating"], prueba["prediccion_ponderada"])
matriz_confusion_no_ponderada = confusion_matrix(prueba["Star Rating"], prueba["prediccion_no_ponderada"])
matriz_confusion_extra = confusion_matrix(prueba["prediccion_ponderada"], prueba["prediccion_no_ponderada"])
# Función para graficar la matriz de confusión
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')
    plt.title(title)
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.show()

# Graficar la matriz de confusión ponderada
plot_confusion_matrix(matriz_confusion_ponderada, "Matriz de Confusión - Ponderada")

# Graficar la matriz de confusión no ponderada
plot_confusion_matrix(matriz_confusion_no_ponderada, "Matriz de Confusión - No Ponderada")

plot_confusion_matrix(matriz_confusion_extra, "Matriz de Confusión - extra")