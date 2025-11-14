
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score 
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import LabelEncoder



# url ="https://raw.githubusercontent.com/andresbaldeon93/Loan_Eligibility_kg/refs/heads/main/Loan%20Eligibility%20Prediction.csv"

# df = pd.read_csv(url,sep=",")
# #print(df.describe())
# #print(df.columns)
# #print(df.dtypes)
# #df.info()
# #print(df.columns)
# #print(type(df))

# #PREPROCESSING
# col_numericas_X = ['Dependents','Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term','Credit_History']
# col_categoricas_X = ['Gender','Married','Education','Self_Employed','Property_Area']

# preprocessor_X = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), col_numericas_X),
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), col_categoricas_X)
#     ],
#     remainder='passthrough'
# )

# le = LabelEncoder()

# #CREACION DE MUESTRAS
# X , y = df.drop(columns=['Customer_ID','Loan_Status'],axis=1),df['Loan_Status']

# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=33)

# #PREPROCESSING
# y_train_index = y_train.index
# y_test_index = y_test.index

# X_train_processed = preprocessor_X.fit_transform(X_train)
# X_test_processed = preprocessor_X.transform(X_test)

# feature_names = preprocessor_X.get_feature_names_out()
# X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
# X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

# y_train_processed = le.fit_transform(y_train)
# y_test_processed = le.transform(y_test)

# y_train_processed_df = pd.DataFrame(y_train_processed,index=y_train_index,columns=['Loan_Status'])
# y_test_processed_df = pd.DataFrame(y_test_processed,index=y_test_index,columns=['Loan_Status'])


# pd.set_option('display.max_columns', None)

# #print(X_train_processed_df)
# #print(y_test_processed_df)

# #MODELAMIENTO

# from sklearn import neighbors
# knn = neighbors.KNeighborsClassifier(n_neighbors=2)
# knn.fit(X_train_processed_df, y_train_processed)
# y_pred = knn.predict_proba(X_test_processed_df)

# knn.score(X_test_processed_df,y_test_processed)
# accuracy_score(y_test_processed, y_pred)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer # Importación crucial para manejar NaN

# Algoritmos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# --- PASO 1: Carga, Limpieza y Separación de Datos ---
# URL usada en tu código anterior
url = "https://raw.githubusercontent.com/andresbaldeon93/Loan_Eligibility_kg/refs/heads/main/Loan%20Eligibility%20Prediction.csv"
df = pd.read_csv(url)

# 1. Codificar la variable target 'Loan_Status' (Y -> 1, N -> 0)
le = LabelEncoder()
df['Loan_Status_Encoded'] = le.fit_transform(df['Loan_Status'])

# 2. Definir columnas y separar X e y
# ID y Target original se eliminan de X
X = df.drop(columns=['Customer_ID', 'Loan_Status', 'Loan_Status_Encoded'], axis=1)
y = df['Loan_Status_Encoded']

# Definición de columnas para el ColumnTransformer
numerical_features = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount']
# Se incluye 'Loan_Amount_Term' como categórica
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                        'Property_Area', 'Credit_History', 'Loan_Amount_Term']

# 3. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- PASO 2: Definición del Preprocesamiento CORREGIDO (CON IMPUTACIÓN) ---

# Pipeline 1: Numéricas (Imputación con media + Estandarización)
numerico_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline 2: Categóricas (Imputación con más frecuente + One-Hot Encoding)
categorico_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer final
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerico_pipeline, numerical_features),
        ('cat', categorico_pipeline, categorical_features)
    ],
    remainder='drop' # Ignorar cualquier otra columna no especificada (ej. IDs)
)

# --- PASO 3: Evaluación de Múltiples Modelos (Baseline) ---
models = {
    "Logistic Regression (LR)": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "Random Forest (RF)": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

print("--- Accuracy de Modelos Base ---")
print(pd.Series(results).sort_values(ascending=False).to_markdown(numalign="left", stralign="left", floatfmt=".4f"))


# --- PASO 4: Optimización de Parámetros (Logistic Regression) ---
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=42, max_iter=1000))])

param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs']
}

# n_jobs=1 para estabilidad en este entorno
grid_search = GridSearchCV(logreg_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0)
grid_search.fit(X_train, y_train)

# --- PASO 5: Resultados Finales ---
best_params = grid_search.best_params_
accuracy_tuned = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))

print("\n--- Resultado de la Optimización (Tuning) ---")
print(f"Mejores Parámetros (LR): {best_params}")
print(f"Accuracy del Modelo Tuned (LR): {accuracy_tuned:.4f}")


# El preprocessor (manejo de faltantes, escalado y codificación) ya está definido.

# Definir la pipeline incluyendo el preprocesador y el clasificador SVC
svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(random_state=42))])

# Definir el grid de parámetros para SVC
param_grid_svc = {
    # C: Fuerza de regularización
    'classifier__C': [0.1, 1, 10], 
    # kernel: Tipo de función del núcleo
    'classifier__kernel': ['linear', 'rbf'], 
    # gamma: Coeficiente del kernel para 'rbf' o 'poly' (scale es el default)
    'classifier__gamma': ['scale', 0.1, 1] 
}

# Inicializar GridSearchCV (n_jobs=1 para evitar errores de paralelización)
grid_search_svc = GridSearchCV(svc_pipeline, param_grid_svc, cv=5, scoring='accuracy', n_jobs=1, verbose=0)

# Ejecutar la búsqueda en los datos de entrenamiento
grid_search_svc.fit(X_train, y_train)

# --- Impresión de Resultados ---
best_params_svc = grid_search_svc.best_params_
accuracy_tuned_svc = accuracy_score(y_test, grid_search_svc.best_estimator_.predict(X_test))

print("\n--- Resultados del Tune de SVC ---")
print(f"Mejor Accuracy (Validación Cruzada): {grid_search_svc.best_score_:.4f}")
print(f"Mejores Parámetros: {best_params_svc}")
print(f"Accuracy en el Conjunto de Prueba (Modelo Tuned): {accuracy_tuned_svc:.4f}")