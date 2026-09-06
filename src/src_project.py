

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

# Modulo para ocultar las advertencias

import warnings
warnings.filterwarnings('ignore')

"""# 1- Carga de datos"""

df = pd.read_csv('/content/drive/MyDrive/Customer Churn Prediction/customer_churn_data.csv')

"""# 2- Análisis exploratorio de Datos"""

df.head()

# Campo, tipo de dato y nulos.

df.info()

# Nulos

nulos = df.isna().sum()
nulos[nulos>0]

df.InternetService.unique()

"""Dejaremos el campo vacío en los registros del campo 'InternetService', que sean 'nan'. Son de clientes que no tienen el servicio de internet contratado."""

# Imputaremos los registros nulos del campo InternetService un campo vacío.

df.InternetService = df['InternetService'].fillna('')

# Validamos los nulos otra vez

nulos = df.isna().sum()
nulos[nulos>0]

# Verificando si tenemos duplicados

df.duplicated().sum()

df.head()

# Estadísticas generales.

df.describe()

# Variable de columnas númericas y su correlacion

cols_numericas = df.select_dtypes(include=["number"])

sns.heatmap(cols_numericas.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.show()

# Verficiamos los porcentajes (Si/No) de Churn

df['Churn'].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.title('Churn (Si/No) %')
plt.show()

df.head(3)

df.groupby('Churn')['Churn'].count()

# Churn agrupado por la media de cargos mensuales

df.groupby('Churn')['MonthlyCharges'].mean()

# Churn, Genero X cargos mensuales

df.groupby(['Churn','Gender'])['MonthlyCharges'].mean()

# Churn por tenencia

df.groupby('Churn')['Tenure'].mean()

# Churn por edad

df.groupby('Churn')['Age'].mean()

# Churn por cargos totales

df.groupby('Churn')['TotalCharges'].mean()

"""Podemos visualizar que el tipo de contrato que más genera es el de 'Mes a mes', era de esperarse ya que los clientes normalmente optan por este tipo de contratos rápidos y no del costo contundente en el momento, como son los demás planes de servicio."""

# Tipo de contrato por cargos mensuales

df.groupby('ContractType')['MonthlyCharges'].mean().plot(kind='bar')
plt.title('Precio $ medio por tipo de contrato')
plt.xlabel('Tipo de contrato')
plt.ylabel('Precio medio')
plt.xticks(rotation=45)
plt.show()

"""Se visualiza que, muchos clientes abandonan un poco rápido el servicio, esto lo podemos visualizar con claridad en el histograma de la tenencia, esa cola larga a la derecha indica que, hay poquísimos clientes que apenas duran 2 años (solo un 27% de los clientes, duran 2 o más años con el servicio) contratando el servicio. Es decir, el servicio ofrecido da a demostrar que no retiene a los clientes por mucho tiempo."""

fig, axes = plt.subplots(1,2, figsize=(12,6))

sns.histplot(df['MonthlyCharges'], bins=15, ax=axes[0])
axes[0].set_title('Histograma de cargos mensuales')
axes[0].set_xlabel('Cargos mensuales')

sns.histplot(df['Tenure'], ax=axes[1])
axes[1].set_title('Histograma de tenencia')
axes[1].set_xlabel('Tenencia (duración que el cliente dura con el servicio)')
plt.show()

print()
print(f'Cantidad de clientes que han durado 2 o más años de servicio: {df[df['Tenure'] > 24].value_counts().sum()} (apróximadamente el 27% de los clientes totales)')

df.head()

"""# 3- Separación de variables X,Y"""

y = df.Churn
X = df[['Age','Gender','Tenure','ContractType','InternetService','MonthlyCharges']]

y.head()

X.head()

"""# 4- Train Split (X,y)"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""# 5- Encoding"""

# Genero | X
# Mujer = 1 | Hombre = 0

X_train.Gender = X_train['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
X_test.Gender = X_test['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Churn | y
# Yes = 1 | No = 0
y_train = y_train.apply(lambda x: 1 if x == 'Yes' else 0)

y_test = y_test.apply(lambda x: 1 if x == 'Yes' else 0)

# One-Hot Encoding a ContractType y InternetService

ohe = OneHotEncoder(sparse_output=False, drop='first', dtype=int)

encoded_train = ohe.fit_transform(X_train[['ContractType', 'InternetService']])
encoded_test = ohe.transform(X_test[['ContractType','InternetService']])

df_enc_train = pd.DataFrame(encoded_train, columns=ohe.get_feature_names_out(), index=X_train.index)
df_enc_test = pd.DataFrame(encoded_test, columns=ohe.get_feature_names_out(), index=X_test.index)

X_train = pd.concat([X_train, df_enc_train], axis='columns')
X_train = X_train.drop(columns=['ContractType','InternetService'], axis='columns')

X_test = pd.concat([X_test, df_enc_test], axis='columns')
X_test = X_test.drop(columns=['ContractType','InternetService'], axis='columns')

X_train.head()

"""# 6- Escalado de datos

Procedemos a escalar los datos, ya que tenemos campos, los cuales tienen rangos de diferentes magnitudes a comparación de otros (una vez el OHE aplicado, lo cual se hizo en la anterior sección), como por ejemplo: Genero, ContracType (OHE), etc. Esto hará que estén en un rango similar entre todos, es decir, evita el sesgo; hace que un '1' (de genero por ejemplo) el modelo no interprete que sea menor que un 100 (de MonthlyCharges), si no que tengan un rango similar.
"""

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

t_s = pd.DataFrame(X_train_scaled)
t_s.describe()

# Almacenamos la instancia:

import joblib
joblib.dump(scaler, 'scaler.pkl')

"""# 7- Random Forest Classifier"""

# Librerias para métricas de rendimiento de los modelos

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

from sklearn.ensemble import RandomForestClassifier

# Instancia del modelo
rf = RandomForestClassifier()

# Hiper-parametros
param_grid_rf = {
    'n_estimators': [32, 64, 128, 256],
    'max_features': [2,3,4],
    'bootstrap': [True, False]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)

grid_rf.fit(X_train_scaled, y_train)

y_pred_rf = grid_rf.predict(X_test_scaled)
y_pred_rf_train = grid_rf.predict(X_train_scaled)

# Mejores parametros del modelo
grid_rf.best_params_

"""De momento, validaremos su Accuracy y demás, con el set de entrenamiento y también el de test para ver si el modelo tiene overfitting"""

print('='*100)
print('TRAIN')
print('='*100)
print(classification_report(y_train, y_pred_rf_train))

print('='*100)
print('TEST')
print('='*100)
print(classification_report(y_test, y_pred_rf))

"""Como hemos dectetado overfitting (aparentemente es que haya profundizado demasiado en los arboles y aprendió los patrones exactos de los datos de entrenamiento, ya que los predice todos tal cual, sin margen de error) en el modelo de Random Forest, vamos a proceder a ajustar sus hiper-parametros y, aplicar el 'balance' de las clases, que como ya detectamos anteriormente (en el EDA) que hay un desbalance claro con los clientes que ABANDONAR y con los que no ABANDONARON, este ajuste hará que el modelo tenga más peso en la clase minorista."""

balance_clases = df.groupby('Churn')['Churn'].count()
balance_clases

from sklearn.metrics import make_scorer, f1_score

m_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
f1_clase_0 = make_scorer(f1_score, pos_label=0)

entrenamiento = []

def rf_ajustado(modelo, X_entrenamiento, X_prueba, y_entrenamiento):

  parametros = {
      'n_estimators': [100,200],
      'max_depth': [3,4,5],
      'min_samples_split': [5, 10, 20],
      'min_samples_leaf': [2, 5, 20],
      'max_features': ['sqrt','log2']
  }

  # GridSearchCV con 5 k-folds
  grid_rf_ajustado = GridSearchCV(modelo, parametros, cv=5, scoring=f1_clase_0)

  # Entrenamiento del modelo
  grid_rf_ajustado.fit(X_entrenamiento, y_entrenamiento)

  print(f'Mejores hiperparametros: {grid_rf_ajustado.best_params_}')

  # Prediccion de TEST
  predict_test = grid_rf_ajustado.predict(X_prueba)

  return grid_rf_ajustado

grid_rf_ajustado = rf_ajustado(modelo=m_rf, X_entrenamiento=X_train_scaled, X_prueba=X_test_scaled,
            y_entrenamiento=y_train)

"""Luego de balancear el modelo y ajustar sus hiper-parametros a unos más limitados, ya el modelo no presenta overfitting y las métricas de rendimiento están mucho mejor que el modelo anterior el cual no está balanceado... si bien las métricas mejoraron mucho y ya no hay overfitting, tenemos una pequeña 'brecha' en la **Precision** en la cual el modelo podrá clasificar una clase (de la clase minorista, los que no abandonan), en cambio por un **Recall** muy bueno."""

print("--- MÉTRICAS EN PRUEBA ---")
print(classification_report(y_test, grid_rf_ajustado.predict(X_test_scaled)))

"""# 8- Modelo de Regresión Logistica"""

log_model = LogisticRegression(class_weight='balanced', random_state=42)

#Entrenamiento
log_model.fit(X_train_scaled, y_train)

# Prediccion con datos de prueba
y_pred_lr = log_model.predict(X_test_scaled)

# Predicción con datos de entrenamiento
y_pred_lr_train = log_model.predict(X_train_scaled)

"""Las métricas de rendimiento arrojan buenos resultados para el modelo. Un Precision de 0.43 y un Recall de 1.00 (en NO) está muy bien, esto aplicandole claro está la clase de 'balanced' al modelo."""

print('===== TRAIN ======')
print(classification_report(y_train, y_pred_lr_train))

print('===== TEST ======')
print(classification_report(y_test, y_pred_lr))

"""# 9- Modelo con KNE

Con KNE, no podemos aplicar 'class_weight=balanced', con este modelo vamos a aplicar 'SMOTE', algoritmo para equilibrar un o desbalanceo en las **clases**: una clase con un peso muchísimo mayor a otro. Este algoritmo nos ayudará a equilibrar ese factor.
"""

# Instancia de smote
smote = SMOTE(random_state=42)

# Versión balanceada de los datos
X_train_balanceado, y_train_balanceado = smote.fit_resample(X_train_scaled, y_train)

# Verificación de las clases balanceadas
print(f'Distribución original:{y_train.value_counts(normalize=True).to_dict()}')
print(f'Distribución balanceada: {y_train_balanceado.value_counts(normalize=True).to_dict()}')

# Instancia del modelo
kne = KNeighborsClassifier()

# Ajuste de hiperparametros
param_grid_kne = {
    'n_neighbors': [3,5,7,9],
    'weights': ['uniform', 'distance']
}

grid_kne = GridSearchCV(kne, param_grid_kne, cv=5, scoring=f1_clase_0)

# Entrenamiento con los datos balanceados con SMOTE
grid_kne.fit(X_train_balanceado, y_train_balanceado)

y_pred_kne = grid_kne.predict(X_test_scaled)

"""Hemos obtuvido las mejores métricas con el modelo de KNE, con un Accuracy(0.91) y un Recall y Precision mejor a los modelos de RLogistica y RF."""

print(classification_report(y_test, y_pred_kne))

"""# 10- Funciones para hacer predicciones con los modelos"""

def prediccion_Encoding(
    Age,
    Gender,
    Tenure,
    MonthlyCharges,
    ContractType_One_Year,
    ContractType_Two_Year,
    InternetService_DSL,
    InternetService_FiberOptic,
    columnas_modelo,
    scaler,
    modelo

):

  campos = pd.DataFrame({
      'Age': [Age],
      'Gender': [Gender],
      'Tenure': [Tenure],
      'MonthlyCharges': [MonthlyCharges],
      'ContractType_One_Year': [ContractType_One_Year],
      'ContractType_Two_Year': [ContractType_Two_Year],
      'InternetService_DSL': [InternetService_DSL],
      'InternetService_FiberOptic': [InternetService_FiberOptic]
  })

  # Reordenamos las filas con las que se entrenó el modelo exactamente

  campos = campos.reindex(columns=columnas_modelo, fill_value=0)

  # Aplicamos escalado

  campos_escalados = scaler.transform(campos)

  # Predicción

  resultado = modelo.predict(campos_escalados)
  return resultado[0]

p =  prediccion_Encoding(Age=40, Gender=0, Tenure=15, MonthlyCharges=65, ContractType_One_Year=0, ContractType_Two_Year=1,
           InternetService_DSL=0, InternetService_FiberOptic=1, columnas_modelo=X_train.columns.tolist(), scaler=scaler, modelo=log_model)

resultado_predict = 'Cliente abandona' if p == 1 else 'Cliente no abandona'
print(f'Predicción: {resultado_predict}')

def prediccion_No_Encoding(
    Age,
    Gender,
    Tenure,
    MonthlyCharges,
    ContractType,
    InternetService,
    columnas_modelo,
    scaler,
    modelo,
    enc
):

    campos = pd.DataFrame({
        'Age': [Age],
        'Gender': [Gender],
        'Tenure': [Tenure],
        'MonthlyCharges': [MonthlyCharges],
        'ContractType': [ContractType],
        'InternetService': [InternetService]
    })

    # Encoding de Género
    campos['Gender'] = campos['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    columnas_cat = ['ContractType', 'InternetService']

    # One-Hot Encoding
    encoded = enc.transform(campos[columnas_cat])
    df_enc = pd.DataFrame(encoded, columns=enc.get_feature_names_out(columnas_cat))

    df_numerico = campos.drop(columns=columnas_cat)
    df_ready = pd.concat([df_numerico, df_enc], axis='columns')

    # Forzar el orden y cantidad de columnas correctas
    df_final = df_ready.reindex(columns=columnas_modelo, fill_value=0)
    df_final = df_final[columnas_modelo]

    # Escalado y Predicción
    campos_escalados = scaler.transform(df_final)
    prediccion = modelo.predict(campos_escalados)

    return prediccion

prediccion_No_Enc = prediccion_No_Encoding(Age=50, Gender='Male',Tenure=10, MonthlyCharges=80, ContractType='One-Year',
                                           InternetService='DSL', columnas_modelo=X_train.columns.to_list(), modelo=grid_rf_ajustado, scaler=scaler, enc=ohe)

resultado_predict = 'Cliente abandona' if prediccion_No_Enc == 1 else 'Cliente no abandona'
print(f'Predicción: {resultado_predict}')

"""# Accuracy/Rendimiento de los modelos

Luego de analizar el accuracy de cada modelo, podemos detectar que el que más tiene y 'mejor' es **Random Forest**, modelo potente por su alto rendimiento y fuerza bruta, con una predicción que gana por mayoría de votos, es muy bueno, pero tiende a sobreaajustar demasiado el modelo, y quedando corto en predicciones con, logicamente, datos nuevos... pero, de todas formas, en este proyecto, buscamos que las predicciones de que un cliente aún continue, sean confiables y que esté balanceada de manera similar a los que abandonan, por ello, seguiremos en la próxima sección verificando las otras métricas de evaluación (Precision, Recall, etc) para que con esas métricas, determinar si realmente el mejor modelo del proyecto es Random Forest, según su score (0.96).
"""

def model_performance(prediccion):
 return accuracy_score(y_test, prediccion)

modelos = [log_model, grid_kne, grid_rf_ajustado] # Modelos

resultados = [] # Acá se almacenará el nombre y el accuracy del modelo seleccionado

for modelo in modelos:
  # Bucle para extraer el nombre del modelo (no del GridSearchCV)

  if type(modelo).__name__ == 'GridSearchCV':
    nombre_modelo = type(modelo.estimator).__name__
  else:
    nombre_modelo = type(modelo).__name__

  # Accuracy
  rendimiento = model_performance(modelo.predict(X_test_scaled))
  # Guardamos en 'resultados'
  resultados.append((rendimiento, nombre_modelo))
resultados.sort(reverse=True)

for rendimiento, nombre_modelo in resultados:
  print(f'{nombre_modelo}: {rendimiento}')

grid_rf_ajustado

"""Almacenamos el objeto ya entrenado de nuestros modelos en .pkl"""

def guardar_modelos(rf=None, lr=None, knn=None):
    if rf is not None:
        joblib.dump(rf, 'rf.pkl')
        print("¡Archivo rf.pkl guardado con éxito!")
    if lr is not None:
        joblib.dump(lr, 'lr.pkl')
        print("¡Archivo lr.pkl guardado con éxito!")
    if knn is not None:
        joblib.dump(knn, 'knn.pkl')
        print("¡Archivo knn.pkl guardado con éxito!")

guardar_modelos(rf=grid_rf_ajustado, lr=log_model, knn=grid_kne)

"""# Precisión, Recall"""

def evaluacion_modelos(y_real, dicc_predicciones):

  """Evaluamos múltiples modelos imprimiendo su classification report (Recall, etc)"""

  for nombre, y_pred in dicc_predicciones.items():
    print(f'=== Reporte de Evaluación: {nombre} ===')
    print(classification_report(y_real, y_pred))
    print("\n" + "="*40 + "\n")

mis_predicciones = {
      'Random Forest': grid_rf_ajustado.predict(X_test_scaled),
      'KNeighbors': y_pred_kne,
      'RLogistica': y_pred_lr
  }

evaluacion_modelos(y_test, mis_predicciones)

"""# Gráfico **ROC** y Curva **Precision-Recall**"""

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

fig, axes = plt.subplots(1,2, figsize=(14,6))

modelos_finales = {
    'Regresión Logística': log_model,
    'KNN': grid_kne,
    'Random Forest': grid_rf_ajustado
}

for nombre, modelo in modelos_finales.items():
  RocCurveDisplay.from_estimator(
      modelo, X_test_scaled, y_test, name=nombre, ax=axes[0], pos_label=0
  )
  PrecisionRecallDisplay.from_estimator(
      modelo, X_test_scaled, y_test, name=nombre, ax=axes[1], pos_label=0
  )

# Gráficos

axes[0].plot([0,1], [0,1], linestyle='--', color='gray', label='Azar (AUC.05)')
axes[0].set_title('Curva ROC (clase "No abandona")')
axes[0].legend(fontsize=8)

axes[1].set_title('Curva Precision-Recall (clase "No abandona")')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

