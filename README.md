# ProyectoFinalCiber
Deteccion de anomalias en hojas de votaciones

Sebastian Cosme Benitez - A00399492
Rodolfo Moreno Gutierrez - A00395543


Dataset usado para los numeros:
https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9

Proceso
Inicialmente se descarga el dataset y se anade una categoria mas, "dot", esta categoria  servira para que el algoritmo
no confunda las anomalias por puntos o viceversa, para generar rapido este dataset se toman x imagenes de puntos (dots)
y se realizan 20 cambios por original
Cambios que hace:

Cambia el tamaño del punto aleatoriamente
Lo mueve a distintas posiciones dentro del lienzo
Lo rota entre -30° y 30°
Añade ruido gaussiano
Aplica desenfoque ligero de forma aleatoria
Guarda cada variante como una imagen nueva

Los puntos de obtienen directamente desde la muestra a analizar (mesas de votacion senado Icesi 2026):

https://divulgacione14congreso.registraduria.gov.co/departamento/31

Se tomo de manera manual 120 capturas de puntos con la herramienta de Windows Sniping tool


## Modelos de aprendizaje
Se utilizan dos modelos de Scikit-Learn entrenados sobre los vectores de pixeles aplanados (1024 caracteristicas por imagen 32x32):

- **Isolation Forest**: detecta slots de digitos atipicos (anomalias) comparando su perfil de pixeles contra el conjunto de entrenamiento. Un score negativo indica anomalia.
- **Random Forest Classifier**: clasifica cada slot en una de las 11 clases (0-9, dot) y produce probabilidades por clase para el reporte.

El modelo entrenado se guarda en `models/isolation_forest.joblib`.


## Para ejecutar rapido
python run.py --dataset dataset/ --pdfs pdfs/

# Para asegurar un pdf como template
python run.py --dataset dataset/ --pdfs pdfs/ --template Template_E-14.pdf


# PDF
Se proveen 3 pdfs, un template, 1 caso sin fraude y 1 con un fraude**

*El caso de fraude es totalmente falso, se realizo durante el caso solo para probar la eficacia del modelo

*El cambio se realizo en la primera pagina, la seccion de votos incinerados el segundo punto(dot) se le realizo cambios para asimilarse a un seis(6)