
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder



url ="https://raw.githubusercontent.com/andresbaldeon93/Loan_Eligibility_kg/refs/heads/main/Loan%20Eligibility%20Prediction.csv"

df = pd.read_csv(url,sep=",")
#print(df.describe())
#print(df.columns)
#print(df.dtypes)
#df.info()
#print(df.columns)
#print(type(df))

#PREPROCESSING
col_numericas_X = ['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term','Credit_History']
col_categoricas_X = ['Gender','Married','Education','Self_Employed','Property_Area']

preprocessor_X = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), col_numericas_X),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), col_categoricas_X)
    ],
    remainder='passthrough'
)

le = LabelEncoder()

#CREACION DE MUESTRAS
X , y = df.drop(columns=['Customer_ID','Loan_Status'],axis=1),df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=33)

y_train_index = y_train.index
y_test_index = y_test.index

X_train_processed = preprocessor_X.fit_transform(X_train)
X_test_processed = preprocessor_X.transform(X_test)

feature_names = preprocessor_X.get_feature_names_out()
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

y_train_processed = le.fit_transform(y_train)
y_test_processed = le.transform(y_test)

y_train_processed_df = pd.DataFrame(y_train_processed,index=y_train_index,columns=['Loan_Status'])
y_test_processed_df = pd.DataFrame(y_test_processed,index=y_test_index,columns=['Loan_Status'])


pd.set_option('display.max_columns', None)

#print(X_train_processed_df.head())
print(y_test_processed_df.head())




#print(normalized_X)
#print(y_train)
#print(X_test)
#print(y_test)
