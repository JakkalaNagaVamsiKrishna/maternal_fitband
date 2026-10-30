# plotting the graphs of the signals for better visual understanding 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("simulated_sensor_data.csv") 

print("Columns:", df.columns.tolist())
print("Number of samples:", len(df))


numeric_columns = df.select_dtypes(include = "number").columns 

if len(numeric_columns) == 0: 
    print("no numericv columns found, check the csv format") 
else: 
    fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(12, 3*len(numeric_columns)), squeeze = False)

    for ax, column in zip(axes.flatten(), numeric_columns): 
        ax.plot(df.index, df[column])
        ax.set_title(column)
        ax.set_xlabel("Sample Number") 
        ax.set_ylabel("Value")
        ax.grid(True) 

    plt.tight_layout()
    plt.savefig("sensor_signal.png", dpi=150)
    plt.show()