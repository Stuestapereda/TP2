import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import heapq

df = pd.read_csv("reviews_sentiment.csv", sep=';')

# Promedio de palabras de comentarios valorados con una estrella
promedio = df[df["Star Rating"] == 1]["wordcount"].mean()
print(f"Promedio de palabras de comentarios valorados con una estrella: {promedio}")

# Eliminar las filas con columnas NA
df_filtrado = df.dropna()

# Convertir el sentimiento a numérico
df_filtrado['titleSentiment'] = df_filtrado['titleSentiment'].apply(lambda ts: 0 if ts == "negative" else 1)

# Dividir el dataset en entrenamiento y prueba
entrenamiento, prueba = train_test_split(df_filtrado, test_size=0.3, random_state=42)

# Función para calcular la distancia euclidiana
def distancia_euclidiana(datos1, datos2):
    return np.linalg.norm(np.array(datos1) - np.array(datos2), axis=1)

# Modelo KNN optimizado
def knn_model(data, variable_clases, variables_predictoras, k, nuevo_dato, ponderacion=False):
    clases = data[variable_clases].unique()
    nuevo_dato_pre_procesado = [float(nuevo_dato[var]) for var in variables_predictoras]

    # Calcular distancias por clase
    distancias = []
    for clase in clases:
        datos_clase = data[data[variable_clases] == clase][variables_predictoras].values
        dist_clase = distancia_euclidiana([nuevo_dato_pre_procesado], datos_clase)
        distancias.extend([(d, clase) for d in dist_clase])

    # Seleccionar las k distancias más cercanas
    k_distancias = heapq.nsmallest(k, distancias)

    # Contar los vecinos por clase
    dic_clases_mas_cercanas = {clase: 0 for clase in clases}
    for d, clase in k_distancias:
        if ponderacion:
            dic_clases_mas_cercanas[clase] += 1 / (d**2 if d > 0 else 1e-5)  # Ponderar por la distancia
        else:
            dic_clases_mas_cercanas[clase] += 1

    # Seleccionar la clase con más vecinos (desempate aleatorio si es necesario)
    max_valor = max(dic_clases_mas_cercanas.values())
    clases_empate = [clase for clase, valor in dic_clases_mas_cercanas.items() if valor == max_valor]

    return np.random.choice(clases_empate) if len(clases_empate) > 1 else clases_empate[0]

for k in range(1, 20):
    print(f"K: {k}")
    # Predicciones con y sin ponderación
    def generar_predicciones(entrenamiento, prueba, ponderacion):
        return [knn_model(entrenamiento, "Star Rating", ["wordcount", "titleSentiment", "sentimentValue"], k, prueba.iloc[i], ponderacion=ponderacion) for i in range(len(prueba))]

    # Generar y asignar predicciones
    prueba["prediccion_ponderada"] = generar_predicciones(entrenamiento, prueba, ponderacion=True)
    prueba["prediccion_no_ponderada"] = generar_predicciones(entrenamiento, prueba, ponderacion=False)

    # Calcular y mostrar la precisión de las predicciones
    def calcular_precision(prueba, tipo_prediccion):
        precision = precision_score(prueba["Star Rating"], prueba[tipo_prediccion], average='macro')
        print(f"Precisión {tipo_prediccion}: {precision:.4f}")
        return precision

    # Calcular precisión para las predicciones ponderadas y no ponderadas
    precision_ponderada = calcular_precision(prueba, "prediccion_ponderada")
    precision_no_ponderada = calcular_precision(prueba, "prediccion_no_ponderada")

# Función para graficar la matriz de confusión
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')
    plt.title(title)
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.show()

# Generar las matrices de confusión y graficarlas
for tipo, predicciones in [("Ponderada", "prediccion_ponderada"), ("No Ponderada", "prediccion_no_ponderada")]:
    matriz_confusion = confusion_matrix(prueba["Star Rating"], prueba[predicciones])
    plot_confusion_matrix(matriz_confusion, f"Matriz de Confusión - {tipo}")

# Imprimir las matrices de confusión adicionales (predicción ponderada vs no ponderada)
matriz_confusion_extra = confusion_matrix(prueba["prediccion_ponderada"], prueba["prediccion_no_ponderada"])
plot_confusion_matrix(matriz_confusion_extra, "Matriz de Confusión - Ponderada vs No Ponderada")
