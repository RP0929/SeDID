import pickle
import os

def print_exp_data():
    # Define the path of the file.
    file_path = "/home/mrp_929/projects/DiffusionMIA/DiffusionMIA/logs/DDPM_TINY-IN_EPS/exp_data.pkl"

    # Open the file in read mode and load the data.
    with open(file_path, "rb") as f:
        exp_data = pickle.load(f)

    # Print the data.
    for key, value in exp_data.items():
        print(f"{key}: {value}")

print_exp_data()
