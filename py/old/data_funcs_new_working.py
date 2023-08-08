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

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         x = df.iloc[:, 0]
#         y = df.iloc[:, 1]

#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name}")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             x = df.iloc[:, 0]
#             y = df.iloc[:, 1]

#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name}")
#             plt.show()

# def find_useful_data(df, threshold_fraction=0.0185):
#     max_y = df.iloc[:, 1].max()
#     threshold = threshold_fraction * max_y

#     mask = df.iloc[:, 1] > threshold
#     useful_data = df[mask]
#     return useful_data

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         useful_data = find_useful_data(df)
#         x = useful_data.iloc[:, 0]
#         y = useful_data.iloc[:, 1]

#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name}")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             useful_data = find_useful_data(df)
#             x = useful_data.iloc[:, 0]
#             y = useful_data.iloc[:, 1]

#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name}")
#             plt.show()

# def find_useful_data(df, threshold_fraction=0.0):
#     max_y = df.iloc[:, 1].max()
#     threshold = threshold_fraction * max_y

#     mask = df.iloc[:, 1] > threshold
#     useful_data = df[mask]
#     return useful_data

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         useful_data = find_useful_data(df)
#         x = useful_data.iloc[:, 0]
#         y = useful_data.iloc[:, 1]

#         # Find local maxima in the 'y' values
#         maxima_indices = find_local_maxima(y)

#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name}")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             useful_data = find_useful_data(df)
#             x = useful_data.iloc[:, 0]
#             y = useful_data.iloc[:, 1]

#             # Find local maxima in the 'y' values
#             maxima_indices = find_local_maxima(y)

#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name}")
#             plt.show()

####################
# This is the stuff that  kind of works new method.



# def find_local_maxima(y_values, order=5):
#     # Find indices of local maxima in the 'y_values' array
#     maxima_indices = argrelextrema(y_values.to_numpy(), np.greater, order=order)[0]
#     return maxima_indices

# # def find_useful_data(df, threshold_fraction=0.0):
# #     max_y = df.iloc[:, 1].max()
# #     threshold = threshold_fraction * max_y

# #     mask = df.iloc[:, 1] > threshold
# #     useful_data = df[mask]
# #     return useful_data

# def find_jump_start_point(df, threshold_pos=0.1, threshold_pre_peak=10000.0):
#     # Assuming 'df' is a DataFrame with two columns (x and y)
#     df = df.apply(pd.to_numeric, errors='coerce')

#     # Calculate the derivative of 'y' with respect to 'x'
#     y_derivative = df.iloc[:, 1].diff() / df.iloc[:, 0].diff()

#     # Find the index where the derivative changes from positive to negative (peak)
#     pre_peak_found = False
#     for i in range(1, len(y_derivative)):
#         if not pre_peak_found and y_derivative.iloc[i] > threshold_pos:
#             pre_peak_found = True
#         if pre_peak_found and y_derivative.iloc[i - 1] > 0 and y_derivative.iloc[i] < 0 and y_derivative.iloc[i - 1] > threshold_pre_peak:
#             jump_start_index = i-1
#             break
#     else:
#         # If no peak is found, return None
#         return None, None

#     return df.iloc[jump_start_index], jump_start_index

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         x = df.iloc[:, 0]
#         y = df.iloc[:, 1]

#         # Find local maxima in the 'y' values
#         maxima_indices = find_local_maxima(y)

#         # Find the point where the jump starts based on the derivative
#         jump_start_point, jump_start_index = find_jump_start_point(df)

#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#         plt.plot(x.iloc[jump_start_index], y.iloc[jump_start_index], 'go', markersize=5)  # Plot jump start point in green
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name}")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             x = df.iloc[:, 0]
#             y = df.iloc[:, 1]

#             # Find local maxima in the 'y' values
#             maxima_indices = find_local_maxima(y)

#             # Find the point with a large jump
#             jump_start_point, jump_start_index = find_jump_start_point(df)

#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#             plt.plot(x.iloc[jump_start_index], y.iloc[jump_start_index], 'go', markersize=5)  # Plot jump start point in green
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name}")
#             plt.show()

###############################################
# New test for flats instead.

# def find_local_maxima(y_values, order=5):
#     # Find indices of local maxima in the 'y_values' array
#     maxima_indices = argrelextrema(y_values.to_numpy(), np.greater, order=order)[0]
#     return maxima_indices

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         x = df.iloc[:, 0]
#         y = df.iloc[:, 1]

#         # Find local maxima in the 'y' values
#         maxima_indices = find_local_maxima(y)

#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name}")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             x = df.iloc[:, 0]
#             y = df.iloc[:, 1]

#             # Find local maxima in the 'y' values
#             maxima_indices = find_local_maxima(y)

#             # Find the point with a large jump

#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name}")
#             plt.show()

# def find_local_maxima(y_values, order=5):
#     # Find indices of local maxima in the 'y_values' array
#     maxima_indices = argrelextrema(y_values.to_numpy(), np.greater, order=order)[0]
#     return maxima_indices

# def find_flat_points(y_values, window_size=5, threshold=0.25):
#     # Calculate the average of the last 'window_size' points
#     last_window_average = y_values.rolling(window_size, min_periods=1).mean().iloc[-1]

#     # Find indices of points close to the last window average
#     flat_indices = (y_values > last_window_average - threshold) & (y_values < last_window_average + threshold)
#     return flat_indices

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         x = df.iloc[:, 0]
#         y = df.iloc[:, 1]

#         # Find local maxima in the 'y' values
#         maxima_indices = find_local_maxima(y)

#         # Find the points representing flat parts based on the average of the last 'window_size' points
#         flat_indices = find_flat_points(y)

#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#         plt.plot(x.loc[flat_indices], y.loc[flat_indices], 'go', markersize=5)  # Plot flat points in green
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name}")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             x = df.iloc[:, 0]
#             y = df.iloc[:, 1]

#             # Find local maxima in the 'y' values
#             maxima_indices = find_local_maxima(y)

#             # Find the points representing flat parts based on the average of the last 'window_size' points
#             flat_indices = find_flat_points(y)

#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'ro', markersize=5)  # Plot local maxima in red
#             plt.plot(x.loc[flat_indices], y.loc[flat_indices], 'go', markersize=5)  # Plot flat points in green
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name}")
#             plt.show()

##################################
# Another try at derivatives
# #################################
def find_local_maxima(y_values, order=10):
    # Find indices of local maxima in the 'y_values' array
    maxima_indices = argrelextrema(y_values.to_numpy(), np.greater, order=order)[0]
    return maxima_indices

def find_first_substantial_maxima(y_values, threshold=0.5):
    global_maxima_index = y_values.to_numpy().argmax()
    return global_maxima_index

# old # def find_last_substantial_maxima(y_values, threshold=0.5):
# #     reversed_y_values = y_values.iloc[::-1]  # Reverse the data
# #     maxima_indices = find_local_maxima(reversed_y_values)
# #     other_maxima_average = reversed_y_values.iloc[maxima_indices[1:]].mean()

# #     for maxima_index in maxima_indices:
# #         if reversed_y_values.iloc[maxima_index] > other_maxima_average + threshold:
# #             return len(y_values) - 1 - maxima_index

# #     return None

# def find_last_substantial_maxima(y_values, threshold_percentage=0.25):
#     maxima_indices = find_local_maxima(y_values)
#     global_maxima_index = y_values.to_numpy().argmax()
#     global_maxima_value = y_values.iloc[global_maxima_index]

#     for maxima_index in reversed(maxima_indices):
#         maxima_value = y_values.iloc[maxima_index]

#         # Check if the maxima value is greater than a certain percentage of the global maximum
#         if maxima_value > global_maxima_value * threshold_percentage:
#             return maxima_index

#     return None

################################################################
# Working
################################################################
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
            # Check if the distance of the last substantial maximum to the global maximum
            # is greater than the average distance
            return maxima_index
            # distance_to_global_maxima = abs(global_maxima_value - maxima_value)
            # average_distance_to_maxima = abs(global_maxima_value - other_maxima_average)
            # if distance_to_global_maxima < average_distance_to_maxima:
            #     return maxima_index

    return None

# def find_flat_regions(y_values, window_size=5, threshold=0.001):
#     # Calculate the moving average of y_values using a rolling window
#     moving_average = y_values.rolling(window=window_size, center=True).mean()

#     # Calculate the derivative of the moving average
#     y_derivative = moving_average.diff().iloc[1:] / y_values.index.to_series().diff().iloc[1:]

#     # Find flat regions based on the derivative values
#     flat_indices = (abs(y_derivative) < threshold).to_numpy()

#     # Find start and end indices of the flat regions
#     flat_starts = np.where(~flat_indices[:-1] & flat_indices[1:])[0] + 1
#     flat_ends = np.where(flat_indices[:-1] & ~flat_indices[1:])[0]

#     if len(flat_starts) == 0 or len(flat_ends) == 0:
#         return None

def find_flat_regions(y_values, window_size=20, threshold=0.001):
    # Calculate the moving average of y_values using a rolling window
    moving_average = y_values.rolling(window=window_size, center=True).mean()

    # Calculate the derivative of the moving average
    y_derivative = moving_average.diff().iloc[1:] / y_values.index.to_series().diff().iloc[1:]

    # Find flat regions based on the derivative values
    flat_indices = (abs(y_derivative) < threshold).to_numpy()

    # Find start and end indices of the flat regions
    flat_starts = np.where(~flat_indices[:-1] & flat_indices[1:])[0] + 1
    flat_ends = np.where(flat_indices[:-1] & ~flat_indices[1:])[0]

    if len(flat_starts) == 0 or len(flat_ends) == 0:
        return []

    # If the first point is in a flat region, remove it
    if flat_indices[0]:
        flat_starts = flat_starts[1:]

    # If the last point is in a flat region, remove it
    if flat_indices[-1]:
        flat_ends = flat_ends[:-1]

    return list(zip(flat_starts, flat_ends))

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

        # Find all local maxima
        maxima_indices = find_local_maxima(y)

        # # Find flat regions
        # flat_regions = find_flat_regions(y, window_size=20, threshold=0.005)

        # # Sort flat regions based on size (length of each flat spot)
        # if flat_regions:
        #     flat_regions.sort(key=lambda r: r[1] - r[0], reverse=True)

        # # Keep only the two largest flat spots
        # if flat_regions is not None:
        #     flat_regions = flat_regions[:2]
        # else:
        #     flat_regions = []

        # Plot the data and the first and last substantial local maxima in green
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-', markersize=2)
        plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'rx', markersize=5)  # Plot all local maxima as red "x"
        if first_maxima_index is not None:
            plt.plot(x.iloc[first_maxima_index], y.iloc[first_maxima_index], 'go', markersize=5)  # Plot first substantial local maxima in green
        if last_maxima_index is not None:
            plt.plot(x.iloc[last_maxima_index], y.iloc[last_maxima_index], 'go', markersize=5)  # Plot last substantial local maxima in green

        # # Plot flat spots as shaded regions
        # if flat_regions:
        #     for flat_start, flat_end in flat_regions:
        #         plt.axvspan(x.iloc[flat_start], x.iloc[flat_end], facecolor='purple', alpha=0.3)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Line Plot - {file_name}")
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

            # Find all local maxima
            maxima_indices = find_local_maxima(y)

            # Find flat regions
            flat_regions = find_flat_regions(y, window_size=5, threshold=0.01)

            # Sort flat regions based on size (length of each flat spot)
            if flat_regions:
                flat_regions.sort(key=lambda r: r[1] - r[0], reverse=True)

            # Keep only the two largest flat spots
            if flat_regions is not None:
                flat_regions = flat_regions[:2]
            else:
                flat_regions = []

            # Plot the data and the first and last substantial local maxima in green
            plt.figure()
            plt.plot(x, y, marker='o', linestyle='-', markersize=2)
            plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'rx', markersize=5)  # Plot all local maxima as red "x"
            if first_maxima_index is not None:
                plt.plot(x.iloc[first_maxima_index], y.iloc[first_maxima_index], 'go', markersize=5)  # Plot first substantial local maxima in green
            if last_maxima_index is not None:
                plt.plot(x.iloc[last_maxima_index], y.iloc[last_maxima_index], 'go', markersize=5)  # Plot last substantial local maxima in green

            # Plot flat spots as shaded regions
            if flat_regions:
                for flat_start, flat_end in flat_regions:
                    plt.axvspan(x.iloc[flat_start], x.iloc[flat_end], facecolor='purple', alpha=0.3)

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Line Plot - {file_name}")
            plt.show()

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         x = df.iloc[:, 0]
#         y = df.iloc[:, 1]

#         # Find the first substantial local maxima
#         first_maxima_index = find_first_substantial_maxima(y)

#         # Find the last substantial local maxima
#         last_maxima_index = find_last_substantial_maxima(y)

#         # Find all local maxima
#         maxima_indices = find_local_maxima(y)

#         # Find flat regions
#         flat_regions = find_flat_regions(y, window_size=20, threshold=0.005)

#         # Plot the data and the first and last substantial local maxima in green
#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'rx', markersize=5)  # Plot all local maxima as red "x"
#         if first_maxima_index is not None:
#             plt.plot(x.iloc[first_maxima_index], y.iloc[first_maxima_index], 'go', markersize=5)  # Plot first substantial local maxima in green
#         if last_maxima_index is not None:
#             plt.plot(x.iloc[last_maxima_index], y.iloc[last_maxima_index], 'go', markersize=5)  # Plot last substantial local maxima in green

#         # Plot flat spots as shaded regions
#         if flat_regions:
#             for flat_start, flat_end in flat_regions:
#                 plt.axvspan(x.iloc[flat_start], x.iloc[flat_end], facecolor='purple', alpha=0.3)

#             # Keep only the two largest flat spots
#             flat_regions.sort(key=lambda region: region[1] - region[0], reverse=True)
#             largest_flat_regions = flat_regions[:2]

#             # Remove everything before the end of the first flat spot and after the start of the second flat spot
#             if len(largest_flat_regions) > 1:
#                 first_flat_end = largest_flat_regions[0][1]
#                 second_flat_start = largest_flat_regions[1][0]

#                 # Plot only the data between the two flat regions
#                 plt.plot(x.iloc[first_flat_end:second_flat_start + 1], y.iloc[first_flat_end:second_flat_start + 1],
#                          marker='o', linestyle='-', markersize=2, color='b')

#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name} (After Removing Flat Spots)")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             x = df.iloc[:, 0]
#             y = df.iloc[:, 1]

#             # Find the first substantial local maxima
#             first_maxima_index = find_first_substantial_maxima(y)

#             # Find the last substantial local maxima
#             last_maxima_index = find_last_substantial_maxima(y)

#             # Find all local maxima
#             maxima_indices = find_local_maxima(y)

#             # Find flat regions
#             flat_regions = find_flat_regions(y, window_size=5, threshold=0.01)

#             # Plot the data and the first and last substantial local maxima in green
#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             plt.plot(x.iloc[maxima_indices], y.iloc[maxima_indices], 'rx', markersize=5)  # Plot all local maxima as red "x"
#             if first_maxima_index is not None:
#                 plt.plot(x.iloc[first_maxima_index], y.iloc[first_maxima_index], 'go', markersize=5)  # Plot first substantial local maxima in green
#             if last_maxima_index is not None:
#                 plt.plot(x.iloc[last_maxima_index], y.iloc[last_maxima_index], 'go', markersize=5)  # Plot last substantial local maxima in green

#             # Plot flat spots as shaded regions
#             if flat_regions:
#                 for flat_start, flat_end in flat_regions:
#                     plt.axvspan(x.iloc[flat_start], x.iloc[flat_end], facecolor='purple', alpha=0.3)

#                 # Keep only the two largest flat spots
#                 flat_regions.sort(key=lambda region: region[1] - region[0], reverse=True)
#                 largest_flat_regions = flat_regions[:2]

#                 # Remove everything before the end of the first flat spot and after the start of the second flat spot
#                 if len(largest_flat_regions) > 1:
#                     first_flat_end = largest_flat_regions[0][1]
#                     second_flat_start = largest_flat_regions[1][0]

#                     # Plot only the data between the two flat regions
#                     plt.plot(x.iloc[first_flat_end:second_flat_start + 1], y.iloc[first_flat_end:second_flat_start + 1],
#                              marker='o', linestyle='-', markersize=2, color='b')

#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name} (After Removing Flat Spots)")
#         plt.show()

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         x = df.iloc[:, 0]
#         y = df.iloc[:, 1]

#         # Find flat regions
#         flat_regions = find_flat_regions(y, window_size=20, threshold=0.005)

#         plt.figure()

#         # Plot the data between the two flat regions
#         if flat_regions:
#             # Keep only the two largest flat spots
#             flat_regions.sort(key=lambda region: region[1] - region[0], reverse=True)
#             largest_flat_regions = flat_regions[:2]

#             # Remove everything before the end of the first flat spot and after the start of the second flat spot
#             if len(largest_flat_regions) > 1:
#                 first_flat_end = largest_flat_regions[0][1]
#                 second_flat_start = largest_flat_regions[1][0]

#                 # Update DataFrame to contain only the data between the two flat regions
#                 df = df.iloc[first_flat_end:second_flat_start + 1]

#         # Find new local maxima and substantial maxima from the filtered data
#         new_maxima_indices = find_local_maxima(df.iloc[:, 1])
#         new_substantial_first_maxima_indices = find_first_substantial_maxima(df.iloc[:, 1])
#         new_substantial_last_maxima_indices = find_last_substantial_maxima(df.iloc[:, 1])

#         # Plot the data
#         plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', linestyle='-', markersize=2, color='b')

#         # Plot new local maxima in red 'x'
#         plt.plot(df.iloc[new_maxima_indices, 0], df.iloc[new_maxima_indices, 1], 'rx', markersize=5)

#         # Plot new substantial maxima in green
#         plt.plot(df.iloc[new_substantial_first_maxima_indices, 0], df.iloc[new_substantial_first_maxima_indices, 1], 'go', markersize=5)
#         plt.plot(df.iloc[new_substantial_last_maxima_indices, 0], df.iloc[new_substantial_last_maxima_indices, 1], 'go', markersize=5)

#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name} (After Removing Flat Spots and New Maxima)")
#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             x = df.iloc[:, 0]
#             y = df.iloc[:, 1]

#             # Find flat regions
#             flat_regions = find_flat_regions(y, window_size=5, threshold=0.01)

#             plt.figure()

#             # Plot the data between the two flat regions
#             if flat_regions:
#                 # Keep only the two largest flat spots
#                 flat_regions.sort(key=lambda region: region[1] - region[0], reverse=True)
#                 largest_flat_regions = flat_regions[:2]

#                 # Remove everything before the end of the first flat spot and after the start of the second flat spot
#                 if len(largest_flat_regions) > 1:
#                     first_flat_end = largest_flat_regions[0][1]
#                     second_flat_start = largest_flat_regions[1][0]

#                     # Update DataFrame to contain only the data between the two flat regions
#                     df = df.iloc[first_flat_end:second_flat_start + 1]

#             # Find new local maxima and substantial maxima from the filtered data
#             new_maxima_indices = find_local_maxima(df.iloc[:, 1])
#             new_substantial_first_maxima_indices = find_first_substantial_maxima(df.iloc[:, 1])
#             new_substantial_last_maxima_indices = find_last_substantial_maxima(df.iloc[:, 1])

#             # Plot the data
#             plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', linestyle='-', markersize=2, color='b')

#             # Plot new local maxima in red 'x'
#             plt.plot(df.iloc[new_maxima_indices, 0], df.iloc[new_maxima_indices, 1], 'rx', markersize=5)

#             # Plot new substantial maxima in green
#             plt.plot(df.iloc[new_substantial_first_maxima_indices, 0], df.iloc[new_substantial_first_maxima_indices, 1], 'go', markersize=5)
#             plt.plot(df.iloc[new_substantial_last_maxima_indices, 0], df.iloc[new_substantial_last_maxima_indices, 1], 'go', markersize=5)

#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name} (After Removing Flat Spots and New Maxima)")
#             plt.show()



################################################################
# Working above but this is a new test plotting derivative
################################################################
# def find_local_maxima(y_values, order=5):
#     # Find indices of local maxima in the 'y_values' array
#     maxima_indices = argrelextrema(y_values.to_numpy(), np.greater, order=order)[0]
#     return maxima_indices

# def find_first_substantial_maxima(y_values, threshold=0.5):
#     global_maxima_index = y_values.to_numpy().argmax()
#     return global_maxima_index

# def find_last_substantial_maxima(y_values, threshold=0.5):
#     reversed_y_values = y_values.iloc[::-1]  # Reverse the data
#     maxima_indices = find_local_maxima(reversed_y_values)
#     other_maxima_average = reversed_y_values.iloc[maxima_indices[1:]].mean()

#     for maxima_index in maxima_indices:
#         if reversed_y_values.iloc[maxima_index] > other_maxima_average + threshold:
#             return len(y_values) - 1 - maxima_index

#     return None

# def plot_dataframes(dataframes, file_name=None):
#     if file_name is not None and file_name not in dataframes:
#         print(f"DataFrame with name '{file_name}' not found.")
#         return

#     if file_name:
#         df = dataframes[file_name]
#         if df.shape[1] != 2:
#             print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#             return
#         df = df.apply(pd.to_numeric, errors='coerce')

#         x = df.iloc[:, 0]
#         y = df.iloc[:, 1]

#         # Find the first substantial local maxima
#         first_maxima_index = find_first_substantial_maxima(y)

#         # Find the last substantial local maxima
#         last_maxima_index = find_last_substantial_maxima(y)

#         # Calculate the derivative at each point
#         derivative = (y.diff() / x.diff()).iloc[1:]

#         # Plot the data and the first and last substantial local maxima in green
#         plt.figure()
#         plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#         if first_maxima_index is not None:
#             plt.plot(x.iloc[first_maxima_index], y.iloc[first_maxima_index], 'go', markersize=5)  # Plot first substantial local maxima in green
#         if last_maxima_index is not None:
#             plt.plot(x.iloc[last_maxima_index], y.iloc[last_maxima_index], 'go', markersize=5)  # Plot last substantial local maxima in green
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title(f"Line Plot - {file_name}")

#         # Plot the derivative in purple
#         plt.twinx()
#         plt.plot(x.iloc[1:], derivative, 'purple', marker='o', linestyle='-', markersize=2, label='Derivative')
#         plt.ylabel("Derivative")
#         plt.legend(loc='upper right')

#         plt.show()

#     else:
#         for file_name, df in dataframes.items():
#             if df.shape[1] != 2:
#                 print(f"Skipping {file_name}: DataFrame should have exactly two columns (x and y).")
#                 continue

#             df = df.apply(pd.to_numeric, errors='coerce')

#             x = df.iloc[:, 0]
#             y = df.iloc[:, 1]

#             # Find the first substantial local maxima
#             first_maxima_index = find_first_substantial_maxima(y)

#             # Find the last substantial local maxima
#             last_maxima_index = find_last_substantial_maxima(y)

#             # Calculate the derivative at each point
#             derivative = (y.diff() / x.diff()).iloc[1:]

#             # Plot the data and the first and last substantial local maxima in green
#             plt.figure()
#             plt.plot(x, y, marker='o', linestyle='-', markersize=2)
#             if first_maxima_index is not None:
#                 plt.plot(x.iloc[first_maxima_index], y.iloc[first_maxima_index], 'go', markersize=5)  # Plot first substantial local maxima in green
#             if last_maxima_index is not None:
#                 plt.plot(x.iloc[last_maxima_index], y.iloc[last_maxima_index], 'go', markersize=5)  # Plot last substantial local maxima in green
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title(f"Line Plot - {file_name}")

#             # Plot the derivative in purple
#             plt.twinx()
#             plt.plot(x.iloc[1:], derivative, 'purple', marker='o', linestyle='-', markersize=2, label='Derivative')
#             plt.ylabel("Derivative")
#             plt.legend(loc='upper right')

#             plt.show()
