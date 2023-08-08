import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

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

def flip_data_before_lowest_local_maxima(data):
    # Calculate the global average of the data
    global_average = np.mean(data)

    # Find the index of the lowest local maximum in the data
    lowest_local_maxima_index = data.to_numpy().argmin()

    # Create a new array to store the flipped data
    flipped_data = data.copy()

    # Flip the data before the lowest local maximum
    for i in range(lowest_local_maxima_index):
        difference = global_average - flipped_data.iloc[i]  # Use iloc instead of [i]
        flipped_data.iloc[i] = global_average + difference

    return flipped_data

def t2_equation(x, b, A, S, T2):
    return b + A * np.exp(-(x - S) / T2)

def fit_t2_equation(x_values, y_values):
    # Set initial guess for b as the minimum of y_values
    initial_b = min(y_values)

    # Set initial guess for A as the difference between maximum and minimum y_values
    initial_A = max(y_values) - min(y_values)

    # Find the index of the maximum y-value excluding NaN values
    max_index = np.nanargmax(y_values)

    # Use the corresponding x-value at the maximum y-value if it exists, otherwise use the first x-value
    initial_S = x_values.iloc[max_index] if max_index < len(x_values) else x_values.iloc[0]

    initial_T2 = 0.1  # Adjust this initial guess based on your understanding of the data
    initial_guesses = [initial_b, initial_A, initial_S, initial_T2]

    try:
        # Convert Pandas Series to NumPy arrays
        x_values_np = x_values.values
        y_values_np = y_values.values

        # Check if the Series are not empty
        if len(x_values_np) == 0 or len(y_values_np) == 0:
            return None

        # Set initial bounds for the curve_fit function
        bounds = ([initial_b - 0.1, 0, -np.inf, 0], [initial_b + 0.1, np.inf, np.inf, np.inf])

        # Perform the curve fitting using scipy.optimize.curve_fit
        popt, _ = curve_fit(t2_equation, x_values_np, y_values_np, p0=initial_guesses, bounds=bounds)
        return popt
    except RuntimeError:
        return None

# Define the T1 equation: b + M * (1 - 2 * exp(-((x - S) / T1)))
def t1_equation(x, b, M, S, T1):
    return b + M * (1 - 2 * np.exp(-((x - S) / T1)))

# Fitting function for T1 equation
def fit_t1_equation(x_values, y_values):
    initial_b = min(y_values)
    initial_M = max(y_values) - min(y_values)

    # Find the index of the maximum y-value excluding NaN values
    max_index = np.nanargmax(y_values)

    # Use the corresponding x-value at the maximum y-value if it exists, otherwise use the first x-value
    initial_S = x_values.iloc[max_index] if max_index < len(x_values) else x_values.iloc[0]

    initial_T1 = 0.05  # Adjust this initial guess based on your understanding of the data
    initial_guesses = [initial_b, initial_M, initial_S, initial_T1]

    try:
        # Convert Pandas Series to NumPy arrays
        x_values_np = x_values.values
        y_values_np = y_values.values

        # Check if the Series are not empty
        if len(x_values_np) == 0 or len(y_values_np) == 0:
            return None

        # Set initial bounds for the curve_fit function
        # Here we assume that M should be positive, and T1 should be greater than 0
        bounds = ([initial_b - 0.1, 0, -np.inf, 0], [initial_b + 0.1, np.inf, np.inf, np.inf])

        # Perform the curve fitting using scipy.optimize.curve_fit
        popt, _ = curve_fit(t1_equation, x_values_np, y_values_np, p0=initial_guesses, bounds=bounds)
        return popt
    except RuntimeError:
        return None

# Calculate the Root Mean Square Error (RMSE) between the fitted curve and the original data points
def calculate_rmse(y_true, y_predicted):
    return np.sqrt(np.mean((y_true - y_predicted) ** 2))

def plot_single_dataframe(df, file_name=None):
    # Create lists to store the RMSE values for T1 and T2 fits
    t1_rmse_values = []
    t2_rmse_values = []

    # df = dataframes[file_name]
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

    data_shape = determine_data_shape(y, maxima_indices)  # Call the new function to determine the shape

    # Drop small maxima based on the moving average of three consecutive points
    maxima_indices = drop_small_maxima(y, maxima_indices)

    # Determine data shape and flip data before the lowest local maximum for T1 cases
    if data_shape == "T1 (decrease to increase)":
        y_flipped = flip_data_before_lowest_local_maxima(y)
        maxima_indices_flipped = [len(y_flipped) - 1 - idx for idx in maxima_indices]
        maxima_indices_flipped = sorted(maxima_indices_flipped)  # Sort the flipped maxima indices
    else:
        y_flipped = y
        maxima_indices_flipped = maxima_indices

    # Create a 1x2 subplot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the data and the first and last substantial local maxima in the first subplot
    axs[0].plot(x, y_flipped, marker='o', linestyle='-', markersize=2)
    axs[0].plot(x.iloc[maxima_indices], y_flipped.iloc[maxima_indices], 'rx', markersize=5)  # Plot all local maxima as red "x"
    axs[1].plot(x.iloc[maxima_indices], y_flipped.iloc[maxima_indices], 'm*', markersize=5)  # Plot all local maxima as purple "*"
    if first_maxima_index is not None:
        axs[0].plot(x.iloc[first_maxima_index - start_index], y_flipped.iloc[first_maxima_index - start_index], 'go', markersize=5)  # Plot first substantial local maxima in green
    if last_maxima_index is not None:
        axs[0].plot(x.iloc[last_maxima_index - start_index], y_flipped.iloc[last_maxima_index - start_index], 'go', markersize=5)  # Plot last substantial local maxima in green

    # Find the x and y values corresponding to the maxima indices
    x_maxima = x.iloc[maxima_indices]
    y_maxima = y_flipped.iloc[maxima_indices]

    # plt.plot(x_maxima, y_maxima, marker='o', linestyle='--', markersize=2)
    if len(x_maxima) > 1 and len(y_maxima) > 1:
        if data_shape == "T2 (only decreases)":
            # Fit the T2 equation to the data and get the coefficients
            t2_fit_result = fit_t2_equation(x_maxima, y_maxima)
            if t2_fit_result is not None:  # Check if t2_fit_result is not None (i.e., the fit was successful)
                b, A, S, T2 = t2_fit_result

                # Calculate the fitted y values based on the T2 equation and fitted coefficients
                fitted_y_values = t2_equation(x_maxima, b, A, S, T2)

                # Calculate the RMSE for T2 fit
                t2_rmse = calculate_rmse(y_maxima, fitted_y_values)
                t2_rmse_values.append(t2_rmse)

                # Plot the fitted curve for T2 in the first subplot
                axs[1].plot(x_maxima, fitted_y_values, label="Fitted Curve (T2)", linestyle='--', color='orange')
                # axs[1].plot(x_maxima, fitted_y_values, label="Fitted Curve T2 (y = b + A * exp(-(x - S) / T2))", linestyle='--', color='orange')

                # Extract the T2 value from the fit
                extracted_t2 = round(T2*1000,2)

                # Print the extracted T2 value
                # print(file_name,"Extracted T2 value:", extracted_t2, "ms")
            else:
                print(file_name,"T2 Fit Issue")

        elif data_shape == "T1 (decrease to increase)":
            # Fit the T1 equation to the data and get the coefficients
            t1_fit_result = fit_t1_equation(x_maxima, y_maxima)
            if t1_fit_result is not None and t1_fit_result.shape[0] > 0:  # Check if t1_fit_result is not empty
                b, M, S, T1 = t1_fit_result

                # Calculate the fitted y values based on the T1 equation and fitted coefficients
                fitted_y_values = t1_equation(x_maxima, b, M, S, T1)

                # Calculate the RMSE for T1 fit
                t1_rmse = calculate_rmse(y_maxima, fitted_y_values)
                t1_rmse_values.append(t1_rmse)

                # Plot the fitted curve for T1 in the first subplot
                # axs[1].plot(x_maxima, fitted_y_values, label="Fitted Curve T1 (y = b + M * (1 - 2 * exp(-(x - S) / T1)))", linestyle='--', color='orange')
                axs[1].plot(x_maxima, fitted_y_values, label="Fitted Curve (T1)", linestyle='--', color='orange')

                # Extract the T1 value from the fit
                extracted_t1 = round(T1*1000,2)

                # Print the extracted T1 value
                # print(file_name,"Extracted T1 value:", extracted_t1, "ms")
            else:
                print(file_name,"T1 fitting failed. No valid initial guesses or insufficient data for fitting.")

    # Add T1 or T2 time and RMSE as subtitles in a framed box
    if data_shape.startswith("T1"):
        if len(t1_rmse_values) > 0:
            mean_t1_rmse = np.mean(t1_rmse_values)
            extracted_t1_str = f"T1 Time: ({extracted_t1:.2f} +/- {mean_t1_rmse:.2f}) ms"
            axs[1].text(0.5, 0.02, extracted_t1_str, transform=axs[1].transAxes, ha='center', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', lw=0.5))
    elif data_shape.startswith("T2"):
        if len(t2_rmse_values) > 0:
            mean_t2_rmse = np.mean(t2_rmse_values)
            extracted_t2_str = f"T2 Time: ({extracted_t2:.2f} +/- {mean_t2_rmse:.2f}) ms"
            axs[1].text(0.5, 0.02, extracted_t2_str, transform=axs[1].transAxes, ha='center', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', lw=0.5))

    # Set axis labels and titles
    axs[0].set_ylabel("Ch1")
    axs[0].set_title(f"{file_name} (Local Maxima)")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Ch1")
    axs[1].set_title(f"{file_name} (Line of Best Fit)")

    # Set legend
    axs[1].legend(loc='best')

    # Set grid lines
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Set plot style and color
    axs[0].set_prop_cycle(color=['#1f77b4', '#ff7f0e'])
    axs[1].set_prop_cycle(color=['#1f77b4', '#ff7f0e'])

    # Set figure background color
    fig.set_facecolor('white')

    # # Add the equation for the fit line below the figure
    # if len(x_maxima) > 1 and len(y_maxima) > 1:
    #     if data_shape.startswith("T2"):
    #         equation_str = "T2 Fit: y = b + A * exp(-(x - S) / T2)"  # Placeholder for the T2 fit equation
    #         axs[1].text(0.5, -0.25, equation_str, transform=axs[1].transAxes, ha='center', va='top', fontsize=10)
    #     elif data_shape.startswith("T1"):
    #         equation_str = "T1 Fit: y = b + M * (1 - 2 * exp(-(x - S) / T1))"  # Placeholder for the T1 fit equation
    #         axs[1].text(0.5, -0.25, equation_str, transform=axs[1].transAxes, ha='center', va='top', fontsize=10)

    # Set subplot spacing
    plt.tight_layout(pad=2)

    # Show the entire plot with both subplots
    plt.show()

def plot_dataframes(dataframes, file_name=None):
    if file_name is not None and file_name not in dataframes:
        print(f"DataFrame with name '{file_name}' not found.")
        return

    if file_name:
        plot_single_dataframe(dataframes[file_name], file_name)
    else:
        for file_name, df in dataframes.items():
            plot_single_dataframe(df, file_name)