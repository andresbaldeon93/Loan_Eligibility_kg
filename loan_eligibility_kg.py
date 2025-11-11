
import polars as pl
import 

url ="https://raw.githubusercontent.com/andresbaldeon93/Loan_Eligibility_kg/refs/heads/main/Loan%20Eligibility%20Prediction.csv"

df = pl.read_csv(url,separator=",")

df.describe()   

print(df.columns)