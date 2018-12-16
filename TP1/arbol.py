import numpy as np
import pandas as pd

from collections import Counter
import operator

def construir_arbol(instancias, etiquetas):
    # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
    
    # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas)
    
    # Criterio de corte: ¿Hay ganancia?
    if ganancia == 0:
        #  Si no hay ganancia en separar, no separamos. 
        return Hoja(etiquetas)
    else: 
        # Si hay ganancia en partir el conjunto en 2
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
        # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)

        #print('cumplen: ',etiquetas_cumplen)
        #print('no cumplen: ',etiquetas_no_cumplen)

        # Paso recursivo (consultar con el computador más cercano)
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen, etiquetas_cumplen)
        sub_arbol_derecho   = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen)
        # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho
        
        # sólo falta conectarlos con un nodo de decisión:
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)

# Definición de la estructura del árbol. 

class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    #  Por ejemplo, {'Si': 2, 'No': 2}
    def __init__(self, etiquetas):
        self.cuentas = dict(Counter(etiquetas.values.flatten()))
    
    def es_hoja(self):
        return True


class Nodo_De_Decision:
    # Un Nodo de Decisión contiene preguntas y una referencia al sub-árbol izquierdo y al sub-árbol derecho
     
    def __init__(self, pregunta, sub_arbol_izquierdo, sub_arbol_derecho):
        self.pregunta = pregunta
        self.sub_arbol_izquierdo = sub_arbol_izquierdo
        self.sub_arbol_derecho = sub_arbol_derecho
        
    def es_hoja(self):
        return False
        
# Definición de la clase "Pregunta"
class Pregunta:
    def __init__(self, atributo, valor):
        self.atributo = atributo
        self.valor = valor
    
    def cumple(self, instancia):
        # Devuelve verdadero si la instancia cumple con la pregunta
        #return instancia[self.atributo] == self.valor
        return instancia[self.atributo] < self.valor
    
    def __repr__(self):
        return "¿Es el valor para {} menor a {}?".format(self.atributo, self.valor)

def gini(etiquetas):
    impureza = 1
    cant_total = len(etiquetas)
    if cant_total > 0:
        etiquetas_unique = np.unique(etiquetas)
        for clase in etiquetas_unique:
            impureza -= (np.count_nonzero(etiquetas == clase)/cant_total)**2
    
    return impureza

def ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    etiquetas = np.append(etiquetas_rama_izquierda, etiquetas_rama_derecha)
    gini_inicial = gini(etiquetas)
    
    gini_izq = gini(etiquetas_rama_izquierda)
    gini_der = gini(etiquetas_rama_derecha)
    proporcion_izq = etiquetas_rama_izquierda.size / etiquetas.size
    proporcion_der = etiquetas_rama_derecha.size / etiquetas.size

    ganancia_gini = gini_inicial - proporcion_izq * gini_izq - proporcion_der * gini_der
    return ganancia_gini


def partir_segun(pregunta, instancias, etiquetas):
    # Esta función debe separar instancias y etiquetas según si cada instancia cumple o no con la pregunta (ver método 'cumple')
    instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = [],[], [], []
    
    etiquetas_cumplen = etiquetas[pregunta.cumple(instancias)]
    etiquetas_no_cumplen = etiquetas[~pregunta.cumple(instancias)]
    instancias_cumplen = instancias[pregunta.cumple(instancias)]
    instancias_no_cumplen = instancias[~pregunta.cumple(instancias)]
    
    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen

def encontrar_mejor_atributo_y_corte(instancias, etiquetas):
    max_ganancia = 0
    mejor_pregunta = None
    for columna in instancias.columns:
        for valor in set(instancias[columna]):
            # Probando corte para atributo y valor
            pregunta = Pregunta(columna, valor)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
   
            ganancia = ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha)
            
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                mejor_pregunta = pregunta
    return max_ganancia, mejor_pregunta

def predecir(arbol, x_t):
    resultado = None
    if arbol.es_hoja():
        return max(arbol.cuentas.items(), key=operator.itemgetter(1))[0]
    if arbol.pregunta.cumple(x_t):
        resultado = predecir(arbol.sub_arbol_izquierdo, x_t)
    else:
        resultado = predecir(arbol.sub_arbol_derecho, x_t)
    
    return resultado

def predecir_proba(arbol, x_t):
    resultado = None
    if arbol.es_hoja():
        return max(arbol.cuentas.items(), key=operator.itemgetter(1))[0]
    if arbol.pregunta.cumple(x_t):
        resultado = predecir(arbol.sub_arbol_izquierdo, x_t)
    else:
        resultado = predecir(arbol.sub_arbol_derecho, x_t)
    
    return resultado
        
class MiClasificadorArbol(): 
    def __init__(self):
        self.arbol = None
        #self.columnas = ['Cielo', 'Temperatura', 'Humedad', 'Viento']
    
    def fit(self, X_train, y_train):
        #self.arbol = construir_arbol(pd.DataFrame(X_train, columns=self.columnas), y_train)
        x = pd.DataFrame(X_train)
        self.arbol = construir_arbol(x, pd.DataFrame(y_train))
        self.columnas = x.columns
        return self
    
    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            
            prediction = predecir(self.arbol, x_t_df) 
            
            predictions.append(prediction)
        return np.array(predictions)

    
    def predict_proba(self, X_test):
        return np.array([ [1-x, x] for x in self.predict(X_test)])
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy
        
    def get_params(self, deep=True):
        return {}


def imprimir_arbol(arbol, spacing=""):
    if isinstance(arbol, Hoja):
        print (spacing + "Hoja:", arbol.cuentas)
        return

    print (spacing + str(arbol.pregunta))

    print (spacing + '--> True:')
    imprimir_arbol(arbol.sub_arbol_izquierdo, spacing + "  ")

    print (spacing + '--> False:')
    imprimir_arbol(arbol.sub_arbol_derecho, spacing + "  ")
