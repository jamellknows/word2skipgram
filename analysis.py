import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from itertools import chain

def calculate_average(arr):
    if not arr:
        return None  # Handle empty array case to avoid division by zero

    total = sum(arr)
    average = total / len(arr)
    return average
# Function to read last 2 entries, calculate averages, variances, and plot
def process_csv(file_path):
    # Read CSV files into DataFrames
    data = [pd.read_csv(file_path)]
    last_columns = [df.iloc[:, -1].tolist() for df in data]
    second_last_columns = [df.iloc[:, -2].tolist() for df in data]
    # print("Last Column:", last_columns)
    # print("Second Last Column:", second_last_columns)
    avg = []
    for val in last_columns:
        avg.append(val) 
    # variances = [entries.var(axis=1) for entries in last_entries]
    # variances_as_lists = [variance.to_list() for variance in variances]
    var = []
    for val in second_last_columns:
        var.append(val)

    avg_flat = [item for sublist in avg for item in sublist]
    var_flat = [item for sublist in var for item in sublist]
    x = [i for i in range(0, len(avg_flat))]
    titles = ['Pi Softmax', 'Exp Softmax', 'Square Softmax']
    avg_calc = calculate_average(avg_flat)
    var_calc = calculate_average(var_flat)
 
    print(f"Exp Softmax : {avg_calc} : {var_calc}")

    # print(f"avg: {avg_flat}\n")
    # print(f"var: {var_flat}\n")
    # Plot the values, averages, and variances
    plt.subplot(3, 1, 1)
    plt.plot(avg_flat, label='Averages')
    plt.plot(var_flat, label='Variances')
    plt.title(f'Exp Softmax')
    plt.legend()
        


    plt.tight_layout()
    plt.show()
    plt.savefig('Softmax.png')

# Specify the file paths
file_paths = ['center_results.csv', 'original_center_results.csv', 'square_center_results.csv']

# Call the function to process and plot the data

process_csv(file_paths[1])
