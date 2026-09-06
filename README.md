# Proyecto de predicción de Churn con Python y Machine Learning

# 📉 Predicción de Abandono de Clientes (Churn) con Machine Learning

Proyecto de clasificación que predice si un cliente abandonará el servicio, comparando tres modelos (Regresión Logística, KNN y Random Forest) bajo un dataset con fuerte desbalance de clases.

---

## 🎯 Pregunta de negocio

¿Qué clientes tienen mayor probabilidad de abandonar el servicio, y qué tan confiable puede ser un modelo para anticiparlo y priorizar a tiempo las acciones de retención?

---

## 📦 Dataset

1,000 registros de clientes con variables como edad, género, tiempo de permanencia (tenure), tipo de contrato, tipo de servicio de internet, cargos mensuales/totales y soporte técnico, junto con la variable objetivo `Churn` (abandona / no abandona). El dataset presenta un desbalance de clases notable: 88% de los clientes abandona, 12% no.

---

## 🛠️ Herramientas

Python · Pandas · NumPy · Scikit-learn · imbalanced-learn (SMOTE) · Matplotlib · Seaborn

---

## 🔍 Proceso

- Análisis exploratorio (EDA) y detección del desbalance de clases
- División train/test antes de cualquier transformación, para evitar fuga de información
- Encoding (One-Hot) y escalado ajustados solo con train
- Manejo del desbalance con dos técnicas distintas según el modelo: `class_weight='balanced'` (Regresión Logística y Random Forest) y `SMOTE` (KNN)
- Comparación de 3 modelos con tuning de hiperparámetros vía `GridSearchCV`, optimizando F1 de la clase minoritaria (no accuracy)
- Evaluación con Precision, Recall y F1 por clase, y curvas ROC / Precision-Recall para comparar modelos en todos los umbrales de decisión
- Función de predicción con validación de inputs

---

## 📈 Resultados

Random Forest fue el modelo con mejor desempeño: Accuracy de 0.95 y F1 de 0.81 en la clase minoritaria ("no abandona"), detectando correctamente al 96% de los clientes que de verdad no abandonan. Superó a KNN (F1 0.68) y a Regresión Logística (F1 0.60), y también obtuvo el mejor resultado en las curvas ROC (AUC 0.98) y Precision-Recall (AP 0.83), confirmando que su ventaja se sostiene en distintos umbrales de decisión, no solo en el 0.5 por defecto.

El principal reto del proyecto fue el desbalance de clases: con solo 117 registros de la clase minoritaria sobre 1,000 en total, la accuracy por sí sola resultaba engañosa, y fue necesario evaluar con métricas más específicas para elegir el modelo correcto.
