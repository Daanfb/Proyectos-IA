#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Frías Balbuena
# NOMBRE: Daniel
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS:
# NOMBRE:
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

import numpy as np

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 

from carga_datos import X_credito, y_credito, X_iris, y_iris, X_votos, y_votos, X_cancer, y_cancer, X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb


# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


## ---------- 

def particion_entr_prueba(X,y,test=0.20):
    
    clases, cantidad_clases = np.unique(y, return_counts=True)

    # Cantidad de cada clase en el conjunto de test
    cantidades_test = (cantidad_clases * test).astype(int)

    # Indices de cada clase en el conjunto de entrenamiento y test
    indices_entr = []
    indices_test = []

    for clase, cantidad_test_clase in zip(clases, cantidades_test):
        #Obtenemos los indices de la clase y los barajamos
        indices_clase = np.where(y == clase)[0]
        np.random.shuffle(indices_clase)

        # Dividimos los indices en entrenamiento y test
        indices_test.append(indices_clase[:cantidad_test_clase])
        indices_entr.append(indices_clase[cantidad_test_clase:])

    # Concatenamos los indices y lo barajamos para que las clases no estén juntas
    indices_entr = np.concatenate(indices_entr)
    indices_test = np.concatenate(indices_test)
    np.random.shuffle(indices_entr)
    np.random.shuffle(indices_test)

    # Si las etiquetas son strings, las convertimos a numeros
    if(y.dtype == '<U11'):
        # Diccionario con el valor que le corresponde a cada clase
        dicc_clases_numero = {clase: i for i, clase in enumerate(clases)}
        
        # Función encargada de cambiar el string por el número
        funcion_cambia_etiqueta = np.vectorize(lambda x: dicc_clases_numero[x])
        y = funcion_cambia_etiqueta(y)

    # Creamos los conjuntos de entrenamiento y test
    Xe, ye = X[indices_entr], y[indices_entr]
    Xp, yp = X[indices_test], y[indices_test]

    return Xe, Xp, ye, yp

# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


# class NormalizadorStandard():

#    def __init__(self):

#         .....
        
#     def ajusta(self,X):

#         .....        

#     def normaliza(self,X):

#         ......

# 


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


# Por ejemplo:
    
    
# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 



# ------ 

class NormalizadorStandard():
    def __init__(self):
        self.ajustado = False
        self.media = None
        self.std = None
        
    def ajusta(self,X):
        self.ajustado = True
        self.media = np.mean(X, axis=0) #Media de cada columna de X
        self.std = np.std(X, axis=0)    #Desviacion tipica de cada columna de X

    def normaliza(self,X):
        if not self.ajustado:
            raise NormalizadorNoAjustado("El normalizador no ha sido ajustado")
        
        return (X - self.media) / self.std


# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 


# ------ 

class NormalizadorMinMax():

    def __init__(self):
        self.ajustado = False
        self.min = None
        self.max = None
        
    def ajusta(self,X):
        self.ajustado = True
        self.min = np.min(X, axis=0) #Minimo de cada columna de X
        self.max = np.max(X, axis=0) #Maximo de cada columna de X

    def normaliza(self,X):
        if not self.ajustado:
            raise NormalizadorNoAjustado("El normalizador no ha sido ajustado")
        
        return (X - self.min) / (self.max - self.min)



# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        
  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit    
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])



# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------

from scipy.special import expit    

def sigmoide(x):
    return expit(x)

class RegresionLogisticaMiniBatch():

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
                batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam

        self.clases = None
        self.w = None

    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
                    early_stopping=False,paciencia=3):
        
        self.clases = np.unique(y)

        # Agregamos una columna de unos a X para el término independiente
        X = np.hstack((np.ones((X.shape[0],1)), X))

        # Pesos iniciales
        self.w = np.random.uniform(-0.5,0.5, (X.shape[1],))

        if early_stopping:
            paciencia_actual = 0
            mejor_ec = np.inf
        
            if Xv is None or yv is None:
                Xv = np.copy(X)
                yv = np.copy(y)
            else:
                # Agregamos una columna de unos a Xv para el término independiente
                Xv = np.hstack((np.ones((Xv.shape[0],1)), Xv))

        if salida_epoch:
            ec, rendimiento = self._entropia_y_rend(X,y)
            print(f"Inicialmente, en entrenamiento EC: {ec}, rendimiento: {rendimiento}.")

            if(Xv is not None and yv is not None):
                ec, rendimiento = self._entropia_y_rend(Xv,yv)
                print(f"Inicialmente, en validación EC: {ec}, rendimiento: {rendimiento}.")

        for epoch in range(n_epochs):
            mini_batches = self._crea_mini_batches(X, y)

            # Calculamos el rate de la epoca actual
            rate_n = self.rate*(1/(1+epoch)) if self.rate_decay else self.rate         

            # Actualizamos los pesos para cada mini_batch
            for mini_batch in mini_batches:
                X_mini = mini_batch[:, :-1]
                y_mini = mini_batch[:, -1]
                
                probs = sigmoide(np.dot(X_mini, self.w))
                error = y_mini - probs

                #Aplicamos broadcasting para que haga un producto escalar y sumamos los elementos de cada columna
                gradiente = np.sum(error.reshape(-1, 1) * X_mini, axis=0)

                # Actualizamos los pesos
                self.w += rate_n * gradiente

            if salida_epoch:
                ec, rendimiento = self._entropia_y_rend(X,y)
                print(f"Epoch {epoch + 1}, en entrenamiento EC: {ec}, rendimiento: {rendimiento}.")

                if early_stopping: 
                    ec, rendimiento = self._entropia_y_rend(Xv,yv)
                    print(f"         en validación    EC: {ec}, rendimiento: {rendimiento}.")
                    if ec < mejor_ec:
                        mejor_ec = ec
                        paciencia_actual = 0
                    else:
                        paciencia_actual += 1
                        if paciencia_actual == paciencia:
                            print("PARADA TEMPRANA")
                            break
            

    def clasifica_prob(self,ejemplos):
        if self.w is None:
            raise ClasificadorNoEntrenado("El clasificador no está entrenado")

        # Agregamos una columna de unos a ejemplos para el término independiente
        ejemplos = np.hstack((np.ones((ejemplos.shape[0],1)),ejemplos))

        return sigmoide(np.dot(ejemplos, self.w))
    
    def clasifica(self,ejemplo):
        if self.w is None:
            raise ClasificadorNoEntrenado("El clasificador no está entrenado")

        probs = self.clasifica_prob(ejemplo)
        return np.where(probs > 0.5, 1, 0)

    # Función para crear los mini-batches que se usarán en el entrenamiento
    def _crea_mini_batches(self, X, y):

        # Concatenamos los datos de entrada con las etiquetas, de forma que ponemos como última columna, y
        datos = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(datos)

        # Creamos los mini-batches con la función split de numpy donde
        # el segundo parámetro es un array con los índices donde se hará la división
        mini_batches = np.split(datos, range(self.batch_tam, datos.shape[0], self.batch_tam))

        return mini_batches

    def _entropia_y_rend(self, X, y):
        # Probabilidad de que sea de la clase 1
        probs = sigmoide(np.dot(X, self.w))     
        
        # Hay veces que la probabilidad es tan baja que da error a la hora de calcular log (devuelve nan),
        # por lo que se le suma un valor muy pequeño para evitarlo
        cte = 1e-15 
        ec = np.sum(np.where(y==1, -np.log(probs + cte), -np.log(1 - probs + cte)))

        # Clasificación
        y_pred = np.where(probs > 0.5, 1, 0)  
        
        rendimiento = np.sum(y == y_pred) / len(y)

        return ec, rendimiento

# ------------------------------------------------------------------------------



# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================



# Este ejercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328




# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------

def rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5):

    clases = np.unique(y)

    # Lista con los indices de cada clase
    indices = [np.where(y == clase)[0] for clase in clases]

    particiones_clases = []
    for indices_clases in indices:
        # Barajamos los indices
        np.random.shuffle(indices_clases)
        
        # Dividimos los indices en n particiones y lo metemos en la lista particiones_clases
        particiones_clases.append(np.array_split(indices_clases, n))

    rendimientos = []
    for i in range(n):
        particion_test = []
        particion_entr = []

        # Dividimos los indices en particiones de test y entrenamiento
        # Ponemos en particion_test la particion i y en particion_entr el resto
        for particion_clase in particiones_clases:
            particion_test.append(particion_clase[i])
            for j in range(n):
                if j!=i:
                    particion_entr.append(particion_clase[j])

        # Concadenamos las particiones para que sea un vector y lo barajamos
        # para que no estén las clases seguidas
        particion_test = np.concatenate(particion_test)
        particion_entr = np.concatenate(particion_entr)
        np.random.shuffle(particion_test)
        np.random.shuffle(particion_entr)

        regresion = RegresionLogisticaMiniBatch(**params)
        regresion.entrena(X[particion_entr], y[particion_entr])
        rend = rendimiento(regresion, X[particion_test], y[particion_test])
        rendimientos.append(rend)
        print(f"Partición: {i+1}. Rendimiento: {rend}")

    return np.mean(rendimientos)



# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ----------------------------

print("** 5) Aplicando los clasificadores binarios **")

# Votos de congresistas US
Xev_votos,Xp_votos,yev_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=0.2)
Xe_votos,Xv_votos,ye_votos,yv_votos=particion_entr_prueba(Xev_votos,yev_votos,test=0.3)

normst_votos=NormalizadorStandard()
normst_votos.ajusta(Xe_votos)
Xe_votos_n=normst_votos.normaliza(Xe_votos)
Xv_votos_n=normst_votos.normaliza(Xv_votos)
Xp_votos_n=normst_votos.normaliza(Xp_votos)

print("\n******** VOTOS DE CONGRESISTAS ********")
print("\n- Validación cruzada con batch_tam=16, rate=0.001, rate_decay=False")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":16,"rate":0.001,"rate_decay":False},
                                Xe_votos_n,ye_votos,n=5)
print("Rendimiento medio:", rend)

print("\n- Validación cruzada con batch_tam=32, rate=0.01, rate_decay=True")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":32,"rate":0.015,"rate_decay":True},
                                Xe_votos_n,ye_votos,n=5)
print("Rendimiento medio:", rend)

print("\n- Validación cruzada con batch_tam=64, rate=0.015, rate_decay=True")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":64,"rate":0.015,"rate_decay":True},
                                Xe_votos_n,ye_votos,n=5)
print("Rendimiento medio:", rend)

print("\nModelo con batch_tam=64, rate=0.015, rate_decay=True, early_stopping=True. Con conjunto de validación")
regresion = RegresionLogisticaMiniBatch(batch_tam=64,rate=0.015,rate_decay=True)
regresion.entrena(Xe_votos_n,ye_votos, Xv=Xv_votos_n, yv=yv_votos, salida_epoch=True, early_stopping=True)
print("Rendimiento en test:", rendimiento(regresion,Xp_votos_n,yp_votos))

# Cáncer de mama
Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.3)

normst_cancer=NormalizadorStandard()
normst_cancer.ajusta(Xe_cancer)
Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

print("\n******** CÁNCER DE MAMA ********")
print("\n- Validación cruzada con batch_tam=16, rate=0.001, rate_decay=False")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":16,"rate":0.001,"rate_decay":False},
                                Xe_cancer_n,ye_cancer,n=5)
print("Rendimiento medio:", rend)

print("\n- Validación cruzada con batch_tam=32, rate=0.01, rate_decay=True")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":32,"rate":0.01,"rate_decay":True},
                                Xe_cancer_n,ye_cancer,n=5)
print("Rendimiento medio:", rend)

print("\n- Validación cruzada con batch_tam=32, rate=0.03, rate_decay=True")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":32,"rate":0.03,"rate_decay":True},
                                Xe_cancer_n,ye_cancer,n=5)
print("Rendimiento medio:", rend)

print("\nModelo con batch_tam=32, rate=0.03, rate_decay=True, early_stopping=True. Con conjunto de validación")
regresion = RegresionLogisticaMiniBatch(batch_tam=32,rate=0.03,rate_decay=True)
regresion.entrena(Xe_cancer_n,ye_cancer, Xv=Xv_cancer_n, yv=yv_cancer, salida_epoch=True, early_stopping=True)
print("Rendimiento en test:", rendimiento(regresion,Xp_cancer_n,yp_cancer))


# Críticas de películas en IMDB
Xe_imdb,Xv_imdb,ye_imdb,yv_imdb=particion_entr_prueba(X_train_imdb,y_train_imdb,test=0.3)

normst_imdb=NormalizadorStandard()
normst_imdb.ajusta(Xe_imdb)
Xe_imdb_n=normst_imdb.normaliza(Xe_imdb)
Xv_imdb_n=normst_imdb.normaliza(Xv_imdb)
Xp_imdb_n=normst_imdb.normaliza(X_test_imdb)

print("\n******** CRÍTICAS DE PELÍCULAS EN IMDB ********")
print("\n- Validación cruzada con batch_tam=16, rate=0.1, rate_decay=True")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":16,"rate":0.1,"rate_decay":True},
                                Xe_imdb_n,ye_imdb,n=5)
print("Rendimiento medio:", rend)

print("\n- Validación cruzada con batch_tam=16, rate=0.015, rate_decay=False")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":16,"rate":0.015,"rate_decay":False},
                                Xe_imdb_n,ye_imdb,n=5)
print("Rendimiento medio:", rend)

print("\n- Validación cruzada con batch_tam=32, rate=0.01, rate_decay=True")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":32,"rate":0.01,"rate_decay":True},
                                Xe_imdb_n,ye_imdb,n=5)
print("Rendimiento medio:", rend)

print("\n- Validación cruzada con batch_tam=64, rate=0.1, rate_decay=True")
rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                {"batch_tam":64,"rate":0.1,"rate_decay":True},
                                Xe_imdb_n,ye_imdb,n=5)
print("Rendimiento medio:", rend)

print("\nModelo con batch_tam=16, rate=0.01, rate_decay=True, early_stopping=True. Con conjunto de validación")
regresion = RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
regresion.entrena(Xe_imdb_n,ye_imdb, Xv=Xv_imdb_n, yv=yv_imdb, salida_epoch=True, early_stopping=True)
print("Rendimiento en test:", rendimiento(regresion,Xp_imdb_n,y_test_imdb))

# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------


class RL_OvR():

    def __init__(self,rate=0.1,rate_decay=False,batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam

        self.clasificadores_binarios = None

    def entrena(self,X,y,n_epochs=100,salida_epoch=False):
        clases = np.unique(y)

        self.clasificadores_binarios = []
        for clase in clases:
            # Clasificador binario para cada clase
            clasificador = RegresionLogisticaMiniBatch(self.rate, self.rate_decay, self.batch_tam)
            
            # Creamos un vector de etiquetas con 1 para la clase actual y 0 para el resto
            etiquetas = np.where(y == clase, 1, 0)
            
            # Entrenamos el clasificador binario
            clasificador.entrena(X, etiquetas, n_epochs, salida_epoch)
            
            # Guardamos el clasificador binario
            self.clasificadores_binarios.append(clasificador)

    def clasifica(self,ejemplos):
        if self.clasificadores_binarios is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado")

        # Hacemos una lista con las probabilidades que devuelve cada clasificador para los ejemplos
        probabilidades = [clasificador.clasifica_prob(ejemplos) for clasificador in self.clasificadores_binarios]
        
        # Devolvemos los indices con mayor probabilidad, que indicará la clase seleccionada
        # Cada columna indica un ejemplo, asi que tenemos que hacer argmax de cada columna
        return np.argmax(probabilidades, axis=0)

            
# --------------------------------


# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
  
# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])
   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     
  

# -------- 

def codifica_one_hot(X):
    # Obtenemos los elementos únicos de cada columna
    unicos = [np.unique(col) for col in X.T]
    
    # Creamos una matriz de ceros con tantas filas como X y tantas columnas como elementos únicos
    res = np.zeros((X.shape[0], sum(len(u) for u in unicos)))

    # Para cada fila, rellenamos las columnas correspondientes
    # con 1 en las posiciones adecuadas
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k,elem in enumerate(unicos[j]):
                if X[i, j] == elem:
                    res[i, sum(len(u) for u in unicos[:j]) + k] = 1
                    break

    return res



# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------

print("\n** 8.1) Conjunto de datos de la concesión de crédito **")

X_credito_onehot = codifica_one_hot(X_credito)

Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(X_credito_onehot, y_credito)
print("\nOvR - X_credito, y_credito - batch_tam=8, rate=0.1, rate_decay=False")
regresion_ovr = RL_OvR(batch_tam=16, rate=0.01, rate_decay=False)
regresion_ovr.entrena(Xe_credito, ye_credito)
print("Rendimiento en prueba:", rendimiento(regresion_ovr, Xp_credito, yp_credito))

print("\nOvR - X_credito, y_credito - batch_tam=8, rate=0.01, rate_decay=True")
regresion_ovr = RL_OvR(batch_tam=32, rate=0.1, rate_decay=True)
regresion_ovr.entrena(Xe_credito, ye_credito)
print("Rendimiento en prueba:", rendimiento(regresion_ovr, Xp_credito, yp_credito))

print("\nOvR - X_credito, y_credito - batch_tam=8, rate=0.15, rate_decay=True (MODELO ELEGIDO)")
regresion_ovr = RL_OvR(batch_tam=32, rate=0.3, rate_decay=True)
regresion_ovr.entrena(Xe_credito, ye_credito)
print("Rendimiento en prueba:", rendimiento(regresion_ovr, Xp_credito, yp_credito))



# ---------------------------------------------------------
# 7.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------

def obtener_digitos (fichero):

    # Obtenemos las lineas del fichero
    with open(fichero, 'r') as f:
        lineas = [l.rstrip('\n') for l in f.readlines()]
        
    # Creamos un array de numpy con los digitos. 
    # Cada digito ocupa 28 lineas. Unimos las 28 lineas y 
    # cambiamos los espacios por 0 y los demás caracteres por 1
    return np.array([[int(c!=' ') for c in "".join(lineas[i:i+28])] for i in range(0, len(lineas), 28)])


def obtener_etiquetas(fichero):

    with open (fichero, 'r') as f:
        lineas = [int(l.rstrip('\n')) for l in f.readlines()]

    return np.array([num for num in lineas])

print("\n** 7.2) Clasificación de imágenes de dígitos escritos a mano **")

Xe_digitos = obtener_digitos("./datos/digitdata/trainingimages")
ye_digitos = obtener_etiquetas("./datos/digitdata/traininglabels")
Xp_digitos = obtener_digitos("./datos/digitdata/testimages")
yp_digitos = obtener_etiquetas("./datos/digitdata/testlabels")

print("\nOvR - Xe_digitos, ye_digitos - batch_tam=16, rate=0.1, rate_decay=False, n_epochs=50")
regresion_ovr_digitos = RL_OvR(batch_tam=16, rate=0.1, rate_decay=False)
regresion_ovr_digitos.entrena(Xe_digitos, ye_digitos, n_epochs=50)
print("Rendimiento en prueba:", rendimiento(regresion_ovr_digitos, Xp_digitos, yp_digitos))

print("\nOvR - Xe_digitos, ye_digitos - batch_tam=32, rate=0.1, rate_decay=True, n_epochs=50 (MODELO ELEGIDO)")
regresion_ovr_digitos = RL_OvR(batch_tam=32, rate=0.1, rate_decay=True)
regresion_ovr_digitos.entrena(Xe_digitos, ye_digitos, n_epochs=50)
print("Rendimiento en prueba:", rendimiento(regresion_ovr_digitos, Xp_digitos, yp_digitos))



# =========================================================================
# EJERCICIO OPCIONAL PARA SUBIR NOTA: 
#    CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA MULTINOMIAL
# =========================================================================


#  Se pide implementar un clasificador para regresión
#  multinomial logística con softmax (VERSIÓN MINIBATCH), descrito en las 
#  diapositivas 55 a 57 del tema de "Complementos de Aprendizaje Automático". 

# class RL_Multinomial():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica_prob(self,ejemplos):

#        ......
 

#     def clasifica(self,ejemplos):

#        ......
   

 
# Los parámetros tiene el mismo significado que en el ejercicio 7 de OvR. 

# En este caso, tiene sentido definir un clasifica_prob, ya que la función
# softmax nos va a devolver una distribución de probabilidad de pertenecia 
# a las distintas clases. 


# NOTA 1: De nuevo, es muy importante para la eficiencia usar numpy para evitar
#         el uso de bucles for convencionales.  

# NOTA 2: Se recomienda usar la función softmax de scipy.special: 

    # from scipy.special import softmax   
#

    
# --------------------------------------------------------------------

# Ejemplo:

# >>> rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)

# >>> rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# >>> rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

# --------------- 

from scipy.special import softmax

class RL_Multinomial():

    def __init__(self,rate=0.1,rate_decay=False,batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam

        self.w = None
        self.clases = None

    def entrena(self,X,y,n_epochs=100,salida_epoch=False):
        self.clases = np.unique(y)

        # Agregamos una columna de unos a X para el término independiente
        X = np.hstack((np.ones((X.shape[0],1)),X))

        # Matriz con los pesos donde cada columna j es un vector para 
        # calcular el grado de pertenencia a la clase j
        self.w  = np.random.uniform(-0.5, 0.5, (X.shape[1], len(self.clases)))

        if salida_epoch:
                ec, rendimiento = self._entropia_y_rend(X,y)
                print(f"Inicialmente, en entrenamiento EC: {ec}, rendimiento: {rendimiento}.")

        for epoch in range(n_epochs):
            mini_batches = self._crea_mini_batches(X, y)

            # Calculamos el rate de la epoca actual
            rate_n = self.rate*(1/(1+epoch)) if self.rate_decay else self.rate
            
            # Actualizamos los pesos para cada mini_batch
            for mini_batch in mini_batches:
                X_mini = mini_batch[:, :-1]
                y_mini = mini_batch[:, -1]
                
                # Resultado de aplicar softmax a X_mini * w
                res_softmax = softmax(np.dot(X_mini, self.w), axis=1)

                # Codificación one-hot
                # No podemos utilizar la función codifica_one_hot porque en un mini-batch, en el y_mini no hay 3 clases,
                # por tanto, esa función nos devolvería una matriz con 2 columnas y no 3 que es el número total de clases.
                # De forma que no tendría la misma dimensión que res_softmax y daría error al restarlas
                one_hot = np.zeros((X_mini.shape[0], len(self.clases)))
                one_hot[np.arange(X_mini.shape[0]), y_mini.astype(int)] = 1
                error = one_hot - res_softmax

                # Realizamos una multiplicación matricial para obtener el gradiente.
                # error es de tamaño (n,m) donde cada fila indica el error de cada ejemplo y cada columna el error de cada clase
                # X_mini es de tamaño (n,d) donde cada fila es un ejemplo y cada columna una característica
                # Por tanto, el resultado de la multiplicación matricial es de tamaño (m,d)
                # donde cada fila es el gradiente de cada clase para la característica correspondiente.
                # Para que el gradiente sea de tamaño (d,m) y poder actualizar los pesos, se transpone el resultado.
                gradiente = np.dot(error.T, X_mini).T

                # Actualizamos los pesos
                self.w += rate_n * gradiente

            if salida_epoch:
                ec, rendimiento = self._entropia_y_rend(X,y)
                print(f"Epoch {epoch + 1}, en entrenamiento EC: {ec}, rendimiento: {rendimiento}.")

    def clasifica(self,ejemplos):
        if self.w is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado")

        ejemplos = np.hstack((np.ones((ejemplos.shape[0],1)), ejemplos))
        res_softmax = softmax(np.dot(ejemplos, self.w), axis=1)
        return np.argmax(res_softmax, axis=1)

    def _crea_mini_batches(self, X, y):

        # Concatenamos los datos de entrada con las etiquetas
        datos = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(datos)

        # Creamos los mini-batches con la función split de numpy donde
        # el segundo parámetro es un array con los índices donde se hará la división
        mini_batches = np.split(datos, range(self.batch_tam, datos.shape[0], self.batch_tam))

        return mini_batches

    def _entropia_y_rend(self, X, y):
        if self.w is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado")

        res_softmax = softmax(np.dot(X, self.w), axis=1)
        one_hot = np.zeros((X.shape[0], len(self.clases)))
        one_hot[np.arange(X.shape[0]), y.astype(int)] = 1

        # Entropía cruzada
        ec = -np.sum(one_hot * np.log(res_softmax))

        # Rendimiento
        clases_pred = np.argmax(res_softmax, axis=1)
        rend = np.sum(clases_pred==y)/len(y)

        return ec, rend

print("\n ** EJERCICIO OPCIONAL **")

Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)
rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)
rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50, salida_epoch=True)

print("\nClasificador con softmax - rate=0.001, batch_tam=8")
print("Rendimiento de clasificador con softmax para conjunto de entrenamiento:", rendimiento(rl_iris_m,Xe_iris,ye_iris))
print("Rendimiento de clasificador con softmax para conjunto de prueba:", rendimiento(rl_iris_m,Xp_iris,yp_iris))
