
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score 


url ="https://raw.githubusercontent.com/andresbaldeon93/Loan_Eligibility_kg/refs/heads/main/Loan%20Eligibility%20Prediction.csv"

df = pl.read_csv(url,separator=",")
#print(df.describe())
#print(df.columns)

df = pl.DataFrame(df)
#x = df['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
#y = df['Loan_Status'] 

X, y = df[:,:12],df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

normalizer = preprocessing.Normalizer().fit(X_train)

normalized_X = normalizer.transform(X_train)
normalized_X_test = normalizer.transform(X_test)


print(normalized_X)
#print(y_train)
#print(X_test)
#print(y_test)
