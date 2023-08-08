import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

def find_csv_files(directory):
    csv_files = []
    csv_dirs = []
    for root, _, files in os.walk(directory):
        for file in files:
            file = file.upper()
            if file.endswith('.CSV'):
                # csv_files.append(os.path.join(root, file))
                csv_files.append(file)
                csv_dirs.append(os.path.join(root, file))
    return csv_files, csv_dirs

def read_csv_files(csv_files):
    dataframes = {}
    for file_path in csv_files:
        df = pd.read_csv(file_path, header=None)
        file_name = os.path.basename(file_path)
        # Filter rows with only numeric values
        df = df[df.apply(lambda row: pd.to_numeric(row, errors='coerce').notnull().all(), axis=1)]
        dataframes[file_name] = df
    return dataframes

def print_head_of_dataframes(dataframes):
    for file_name, df in dataframes.items():
        print(f"Head of {file_name}:")
        # print(df.head())
        print(df)
        print("\n")

def find_local_maxima(y_values, order=10):
    # Find indices of local maxima in the 'y_values' array
    maxima_indices = argrelextrema(y_values.to_numpy(), np.greater, order=order)[0]
    return maxima_indices

def find_first_substantial_maxima(y_values, threshold=0.5):
    global_maxima_index = y_values.to_numpy().argmax()
    return global_maxima_index

def find_last_substantial_maxima(y_values, threshold_percentage_lower=0.12, threshold_percentage_upper=0.60):
    maxima_indices = find_local_maxima(y_values, round(len(y_values)*0.01))
    global_maxima_index = y_values.to_numpy().argmax()
    global_maxima_value = y_values.iloc[global_maxima_index]

    # Calculate the average of the other maxima values
    other_maxima_average = y_values.iloc[maxima_indices[:-1]].mean()

    for maxima_index in reversed(maxima_indices):
        maxima_value = y_values.iloc[maxima_index]

        # Check if the maxima value is greater than a certain percentage of the global maximum
        if (maxima_value > global_maxima_value * threshold_percentage_lower) and (maxima_value < global_maxima_value * threshold_percentage_upper):
            return maxima_index
    return None

import numpy as np

def determine_data_shape(y_values, maxima_indices, window_size=0.1, threshold_concavity=-5.0):
    if len(maxima_indices) < 2:
        return "Insufficient maxima points for classification"

    # Calculate the start and end indices for the region between the first and last substantial maxima
    start_index = maxima_indices[0]
    end_index = maxima_indices[-1]

    # Slice the y-values to contain only the data between the start and end indices
    y_values_slice = y_values.iloc[start_index:end_index + 1]

    # Check if there are enough points for the rolling average calculation
    if len(y_values_slice) < 2:
        return "Insufficient points for classification"

    # Calculate the rolling average with the specified window size, using only the maxima points
    maxima_indices_relative = [idx - start_index for idx in maxima_indices]  # Adjust maxima indices
    rolling_avg = y_values_slice.iloc[maxima_indices_relative].rolling(window=int(window_size * len(maxima_indices_relative)), min_periods=1).mean()

    # Check if there are enough points for classification
    if len(rolling_avg) < 2:
        return "Insufficient points for classification"

    # Calculate the percentage increase at every point
    percentage_increases = (rolling_avg - rolling_avg.shift(1)) / rolling_avg.shift(1) * 100

    # Calculate the second derivative of the percentage increase
    second_derivative = np.gradient(np.gradient(percentage_increases))
    # print(second_derivative)

    # Check if the second derivative indicates concavity (negative values)
    if any(second_derivative < threshold_concavity):
        return "T1 (decrease to increase)"
    else:
        return "T2 (only decreases)"

def drop_small_maxima(y_values, maxima_indices, threshold_percent=1):
    new_maxima_indices = [maxima_indices[0]]  # Keep the first maxima always

    for i in range(1, len(maxima_indices)-1):
        prev_maxima_index = maxima_indices[i-1]
        curr_maxima_index = maxima_indices[i]
        next_maxima_index = maxima_indices[i+1]

        last_three_values = [
            y_values.iloc[prev_maxima_index],
            y_values.iloc[curr_maxima_index],
            y_values.iloc[next_maxima_index]
        ]

        average_last_three = sum(last_three_values) / len(last_three_values)
        lowest_value = min(last_three_values)
        difference_percent = 100 * abs(y_values.iloc[curr_maxima_index] - lowest_value) / average_last_three

        if difference_percent >= threshold_percent:
            new_maxima_indices.append(curr_maxima_index)

    new_maxima_indices.append(maxima_indices[-1])  # Keep the last maxima always
    return new_maxima_indices


def plot_dataframes(dataframes, file_name=None):
    if file_name is not None and file_name not in dataframes:
        print(f"DataFrame with name '{file_name}' not found.")
        return

    if file_name:
        df = dataframes[file_name]
        if df.shape[1] != 2:
            print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
            return
        df = df.apply(pd.to_numeric, errors='coerce')

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Find the first substantial local maxima
        first_maxima_index = find_first_substantial_maxima(y)

        # Find the last substantial local maxima
        last_maxima_index = find_last_substantial_maxima(y)

        # Calculate the start and end indices for the region to plot
        start_index = max(0, first_maxima_index - 5)
        end_index = min(len(x) - 1, last_maxima_index + 5)

        # Slice the DataFrame to contain only the data between the start and end indices
        df = df.iloc[start_index:end_index + 1]
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Find all local maxima within the region to plot
        maxima_indices = find_local_maxima(y)

        # Determine the shape of the data
        data_shape = determine_data_shape(x, y, maxima_indices)  # Call the new function to determine the shape

        # Drop large maxima points
        # threshold = 0.5  # Adjust this threshold as needed
        # maxima_indices = drop_small_maxima(y, maxima_indices
        # Drop small maxima based on the moving average of three consecutive points
        maxima_indices = drop_small_maxima(y, maxima_indices)

        # Plot the data and the first and last substantial local maxima in green
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-', markersize=2)
        plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'rx', markersize=5)  # Plot all local maxima as red "x"
        if first_maxima_index is not None:
            plt.plot(x.iloc[first_maxima_index - start_index], y.iloc[first_maxima_index - start_index], 'go', markersize=5)  # Plot first substantial local maxima in green
        if last_maxima_index is not None:
            plt.plot(x.iloc[last_maxima_index - start_index], y.iloc[last_maxima_index - start_index], 'go', markersize=5)  # Plot last substantial local maxima in green

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Line Plot - {file_name}")
        plt.text(0.5, 0.95, f"Data Shape: {data_shape}", transform=plt.gca().transAxes, ha='center', va='top', fontsize=12)
        plt.show()

    else:
        for file_name, df in dataframes.items():
            if df.shape[1] != 2:
                print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
                continue

            df = df.apply(pd.to_numeric, errors='coerce')

            x = df.iloc[:, 0]
            y = df.iloc[:, 1]

            # Find the first substantial local maxima
            first_maxima_index = find_first_substantial_maxima(y)

            # Find the last substantial local maxima
            last_maxima_index = find_last_substantial_maxima(y)

            # Calculate the start and end indices for the region to plot
            start_index = max(0, first_maxima_index - 5)
            end_index = min(len(x) - 1, last_maxima_index + 5)

            # Slice the DataFrame to contain only the data between the start and end indices
            df = df.iloc[start_index:end_index + 1]
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]

            # Find all local maxima within the region to plot
            maxima_indices = find_local_maxima(y)

            data_shape = determine_data_shape(y, maxima_indices)  # Call the new function to determine the shape

            # Drop large maxima points
            # maxima_indices = drop_small_maxima(y, maxima_indices)
            # Drop small maxima based on the moving average of three consecutive points
            maxima_indices = drop_small_maxima(y, maxima_indices)

            # Plot the data and the first and last substantial local maxima in green
            plt.figure()
            plt.plot(x, y, marker='o', linestyle='-', markersize=2)
            plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'rx', markersize=5)  # Plot all local maxima as red "x"
            if first_maxima_index is not None:
                plt.plot(x.iloc[first_maxima_index - start_index], y.iloc[first_maxima_index - start_index], 'go', markersize=5)  # Plot first substantial local maxima in green
            if last_maxima_index is not None:
                plt.plot(x.iloc[last_maxima_index - start_index], y.iloc[last_maxima_index - start_index], 'go', markersize=5)  # Plot last substantial local maxima in green

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Line Plot - {file_name}")
            plt.text(0.5, 0.95, f"Data Shape: {data_shape}", transform=plt.gca().transAxes, ha='center', va='top', fontsize=12)
            plt.show()

