import pandas as pd

input_file = "household_power_consumption.txt"
output_file = "dataset/energy_dataset.csv"

chunksize = 100000  # number of rows processed at once

reader = pd.read_csv(input_file, sep=';', chunksize=chunksize, low_memory=False)

for i, chunk in enumerate(reader):
    if i == 0:
        chunk.to_csv(output_file, index=False, mode='w')
    else:
        chunk.to_csv(output_file, index=False, mode='a', header=False)

print("Conversion completed!")
