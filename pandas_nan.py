
import pandas as pd
import numpy as np

# Read csv file into a pandas dataframe
df = pd.read_csv("data/property_data.csv")

# Take a look at the first few rows
print(df.head())

# Looking at the ST_NUM column
print(df['ST_NUM'])
print(df['ST_NUM'].isnull())

# Looking at the NUM_BEDROOMS column
print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull())

# some rows values have unique
# "no data" representations

# Making a list of missing value types
missing_values = ["n/a", "na", "--"]
df = pd.read_csv("data/property_data.csv", na_values = missing_values)

# Looking at the NUM_BEDROOMS column
print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull())

# unexpected datatype. i.e. boolean column
# containing numbers
# Looking at the OWN_OCCUPIED column
print(df['OWN_OCCUPIED'])
print(df['OWN_OCCUPIED'].isnull())

# Detecting numbers and setting to NaN
for i, row in enumerate(df['OWN_OCCUPIED']):
    try:
        int(row)
        df.loc[i, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
print(df['OWN_OCCUPIED'])
print(df['OWN_OCCUPIED'].isnull())

# Any missing values?
print(df.isnull().values.any())

# getting count of NA values in each col
print(df.isnull().sum())

# Total number of missing values
print(df.isnull().sum().sum())

# Replace missing values with a number
df['ST_NUM'].fillna(125, inplace=True)

# Replace using median
median = df['NUM_BEDROOMS'].median()
df['NUM_BEDROOMS'].fillna(median, inplace=True)