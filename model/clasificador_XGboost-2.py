#!/usr/bin/env python
# coding: utf-8

# # clasificador_XGboost-2

# In[1]:


# Reimportar librerías tras el reinicio
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Cargar archivo nuevamente
file_path = "Encuesta Digital Girls_SP&ENG_C_Consolidados.xlsx"
df_new = pd.read_excel(file_path)
df_new.head()


# In[13]:


df_new.count()


# Estos eran los datos sin consolidar.
# 
# NEGOCIOS Y FINANZAS                     106
# TEGNOLOGIA E INFORMATICA                 62
# SALUD Y MEDICINA                         43
# DERECHO Y CIENCIAS SOCIALES              35
# INGENIERIA Y CONSTRUCCION                32
# EDUCACIÓN Y FORMACIÓN                    25
# COMUNICACIÓN Y MEDIOS                    25
# CREATIVAS Y ARTISTICAS                   13
# CIENCIAS CIENTIFICAS E INVESTIGACION      8
# NATURALEZA Y MEDIO AMBIENTE               7
# Name: Label, dtype: int64

# In[4]:


df_new['Label'].value_counts()


# In[5]:


# Eliminamos la columna de ID
df_new.drop(['ID'], axis=1, inplace=True)
df_new.head()


# ## Procesamos el dataset

# In[6]:


# Preprocesamiento
df_model = df_new.copy()
le = LabelEncoder()
df_model['Label'] = le.fit_transform(df_model['Label'])
df_model = pd.get_dummies(df_model, columns=['Asignatura1', 'Asignatura1.1', 'Personalidad'])

X = df_model.drop(columns=['Label'])
y = df_model['Label']


# In[7]:


# Separación
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Limpieza de columnas
X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '_').replace('>', '_') for col in X_train.columns]
X_test.columns = X_train.columns
X_test.columns


# In[8]:


# Entrenar modelo
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)


# In[9]:


# Predicciones
y_pred = model.predict(X_test)


# In[10]:


# Matriz de confusión y Accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
labels = le.inverse_transform(sorted(y_test.unique()))


# In[11]:


# Visualización
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión - XGBoost (Multiclase)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[12]:


accuracy


# In[14]:


import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=10)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




