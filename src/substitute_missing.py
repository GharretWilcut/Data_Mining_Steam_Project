# Substitute missing data with data from steam API

from data_io import get_data

df = get_data()
print(df.size)
print(df.columns)
