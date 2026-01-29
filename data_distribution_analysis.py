import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import csv
import argparse


def extract_data(csv_file_path, data_type=str):
    """Function to read and process cpp_output.csv"""

    data = np.loadtxt(csv_file_path, dtype=str, delimiter=',', skiprows=1)

    keys = ["frameID", "patientID", "age", "sex", "brand", "GT", "algo1", "algo2", "illumination"]
    data_dicts = [dict(zip(keys, row)) for row in data]

    data=[]

    for entry in data_dicts:
        type_value=entry[data_type]
        data.append(entry)
    return data


def data_type_distribution_plot(total_distribution, output_path, data_type):
    """Function to plot distribution of a specific data type"""

    type_counts = {}
    for entry in total_distribution:
        type_value = entry[data_type]
        if type_value in type_counts:
            type_counts[type_value] += 1
        else:
            type_counts[type_value] = 1

    types = list(type_counts.keys())
    counts = list(type_counts.values())

    plt.figure(figsize=(20, 6))
    sns.barplot(x=types, y=counts)
    plt.legend([],[], frameon=False)
    plt.xlabel(data_type)
    plt.ylabel('Count')
    plt.title(f'Distribution of {data_type}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}\\Figs\\subgroup_distribution_{data_type}.png")
    plt.close()


def cohort_analysis(csv_file_path, output_path):
    """Function to analyze cohort based on patientID distribution"""
    
    data = np.loadtxt(csv_file_path, dtype=str, delimiter=',', skiprows=1)

    keys = ["frameID", "patientID", "age", "sex", "brand", "GT", "algo1", "algo2", "illumination"]
    data_dicts = [dict(zip(keys, row)) for row in data]

    distribution_patientID = {}

    for entry in data_dicts:
        patientID_value = entry["patientID"]
        if patientID_value not in distribution_patientID:
            distribution_patientID[patientID_value] = 1  # Initialize count for new patientID
        else:
            distribution_patientID[patientID_value] += 1  # Increment count for existing patientID

    # Convert the dictionary to a list of dictionaries for plotting
    distribution_patientID_list = [
        {"patientID": patientID, "frames": count} for patientID, count in distribution_patientID.items()
    ]

    # # Debug print to check the distribution
    # print(distribution_patientID_list)

    # Plots information about the cohorts of patients: number of frames for each patientID
    plt.figure(figsize=(20, 6))
    patientIDs = [entry["patientID"] for entry in distribution_patientID_list]   
    frames_counts = [entry["frames"] for entry in distribution_patientID_list]
    sns.barplot(x=patientIDs, y=frames_counts)
    plt.legend([],[], frameon=False)
    plt.xlabel('patientID') 
    plt.ylabel('Number of Frames')
    plt.title('Distribution of Frames per patientID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_image_path = f"{output_path}\\Figs\\cohort_analysis_patientID_distribution.png"
    plt.savefig(output_image_path)
    plt.close() 

    

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Data Distribution Analysis")
    argument_parser.add_argument("--csv_file_path", type=str, required=True, help="Path to the CSV file") # '.\\Baldini_Detection_Assignment\\cpp_output.csv'
    argument_parser.add_argument("--output_path", type=str, required=True, help="Folder to save the outputs") # '.\\Baldini_Detection_Assignment'
    argument_parser.add_argument("--data_type", type=str, help="Data type to analyze (e.g., 'sex', 'brand', 'illumination')")
    args = argument_parser.parse_args()     


    distribution = extract_data(args.csv_file_path, args.data_type)
    data_type_distribution_plot(distribution, args.output_path, args.data_type)
    # cohort_analysis(args.csv_file_path, args.output_path)

# TO RUN:
# python .\data_distribution_analysis.py --csv_file_path "C:\Users\cbaldini\Desktop\Baldini_Detection_Assignment\Baldini_Detection_Assignment\cpp_output.csv" --output_path "C:\Users\cbaldini\Desktop\Baldini_Detection_Assignment\Baldini_Detection_Assignment" --data_type "age"