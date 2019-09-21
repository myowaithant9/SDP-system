import pandas as pd
import openpyxl

dataset_path = 'trace.csv'
dataset = pd.read_csv(dataset_path)
print('Dataset')
print(dataset)

print('dataset describe')
print(dataset.describe())
