#import required modules
import pandas as pd
import numpy as np

#create new data frame from CSV for accounts receivable
recs = pd.read_csv("accounts_receivable.csv")
recs["inv_date"] = pd.to_datetime(recs["inv_date"])
recs["due_date"] = pd.to_datetime(recs["due_date"])
recs["paid_date"] = pd.to_datetime(recs["paid_date"])

#create a new column for age of the receivable
recs["age"] = recs["paid_date"] - recs["inv_date"] 

#create a new column for late receivables
recs["late"] = recs["paid_date"] > recs["due_date"]

#display the data to the user
print(recs.dtypes)
print(recs.head())

#describe statistics for score and age
print(" Descriptive statistics for credit score: ")
print(recs["score"].describe())
print(" Descriptive statistics for age: ")
print(recs["age"].dt.days.describe())

#import python library matplotlib
import matplotlib.pyplot as plt

#create a histogram for the age of the receivable when paid (days)
plt.hist(x=recs["age"].dt.days, bins=20)

#label the age of receivables histogram
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age of Receivables when Paid")

#display the age of receivable histogram
print(plt.show())

#create a scatterplot with respective labels
plt.scatter(recs["age"].dt.days, recs["score"])
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age of Receivables when Paid")

#import the scipy model
from scipy.stats import pearsonr
import scipy
import statsmodels.api as sm

#create a correlation and display it
print(" Correlation coeifficient and p-value: ")
print(pearsonr(recs["age"].dt.days, recs["score"]))

#create two variables for linear regression
y = recs["age"].dt.days
x = recs["score"]

#add the constant for linear regression
x = sm.add_constant(x)

#create the linear regression model
mod = sm.OLS(y, x)

#estimate the fit of the linear regression model
results = mod.fit()

#print the results
print(results.summary())






