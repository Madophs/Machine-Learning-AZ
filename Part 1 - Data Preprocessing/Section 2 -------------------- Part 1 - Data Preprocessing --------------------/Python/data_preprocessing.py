import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Let's take care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Let's encode  our data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Transform the the column data, but keeping the rest of the non used data (passthrough)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# We need to work in numpy arrays
X = np.array(ct.fit_transform(X));

# Let's encode our dependent variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(X)
print(y)

# Now let's split our data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Let's scale (standardization) our training data
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train[:, 3:] = s_scaler.fit_transform(X_train[:, 3:])

# Let's ONLY transform our test train set with same results obtained by the training set
X_test[:, 3:] = s_scaler.transform(X_test[:, 3:])

print("X_train: \n", X_train)
print("X_test: \n", X_test)
