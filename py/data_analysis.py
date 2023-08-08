"""
Created by: Matt Myers
Date: 07/29/2023
Class: Quantum Computing Laboratory

NMR Lab help program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Read in CSV file
    2. Find the peaks
    3. Plot the line of best fit for the peaks
        > Flip the peaks for T1
    4. Return T1 or T2 time depending on shape
"""
# Libraries
import os
import data_funcs_new_temp as f

# Truth statements for optional tests
print_csvs = False

# Directory where all the csv files are located
direct = r'D:\Documents\UW-Madison\CourseWork\Summer\Lab\Paper\NMR'
if direct is not None:
    os.chdir(direct)
else:
    os.chdir(os.getcwd())

# Recursively finding all csv files in subdirectories
csv_files_found, csv_paths = f.find_csv_files(direct)
print("_"*24+"\nCSV files found:\n")
for file_path in csv_files_found:
    print(">",file_path)
print("_"*24,"\n")

# Reading CSV files
dataframes = f.read_csv_files(csv_paths)

# Printing the CSVs for testing
if print_csvs:
    f.print_head_of_dataframes(dataframes)

# Plotting CSV files
# f.plot_dataframes(dataframes, file_name="GLYCERIN90_T1.CSV")
# f.plot_dataframes(dataframes, file_name="GLYCERIN90_T2.CSV")
# f.plot_dataframes(dataframes, file_name="MINERAL_T2.CSV")
# f.plot_dataframes(dataframes, file_name="GLYRCERIN100_T2.CSV")
# f.plot_dataframes(dataframes, file_name="MINERL_T1.CSV")
f.plot_dataframes(dataframes)