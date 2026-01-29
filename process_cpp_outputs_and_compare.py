import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import csv
import pandas as pd
from scipy.stats import mannwhitneyu
import argparse


def process_cpp_outputs(csv_file_path):
    """Function to read and process "cpp_output.csv"
    Args:
        csv_file_path: Path to the "cpp_output.csv" file
    Returns:
        X_algo1: Predictions from algo1
        y: Ground truth labels
        X_algo2: Predictions from algo2
    """

    data = np.loadtxt(csv_file_path, dtype=str, delimiter=',', skiprows=1)
    # print("Debug: ", data.shape, data[0])

    keys = ["frameID", "patientID", "age", "sex", "brand", "GT", "algo1", "algo2", "illumination"]
    data_dicts = [dict(zip(keys, row)) for row in data]

    X_algo1 = [] # Predictions from algo1
    y = [] # Ground truth labels
    X_algo2 = [] # Predictions from algo2

    for entry in data_dicts:
        # Convert 'N'/'P' to 0/1, both for GT and predictions
        label = 0 if str(entry["GT"]) == 'N' else 1
        pred_algo1 = 0 if str(entry["algo1"]) == 'N' else 1
        pred_algo2 = 0 if str(entry["algo2"]) == 'N' else 1

        X_algo1.append(pred_algo1)
        y.append(label)
        X_algo2.append(pred_algo2)

    X_algo1 = np.array(X_algo1)
    y = np.array(y)
    X_algo2 = np.array(X_algo2)

    return X_algo1, y, X_algo2


def compute_metrics(X, y, algo_column, output_dir):
    """Function to compute metrics

    Args:
        X: Predictions
        y: Ground truth labels

    Returns:
        fpr: False Positive Rate
        tpr: True Positive Rate
        roc_auc: Area Under the Curve
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
        cf_matrix: Confusion matrix
    """
    # Compute metrics
    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    cf_matrix = confusion_matrix(y, X)
    accuracy = accuracy_score(y, X)
    precision = precision_score(y, X)
    recall = recall_score(y, X)
    f1 = f1_score(y, X)

    results_data = {
        'Algorithm': [algo_column],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }

    results_df = pd.DataFrame(results_data)

    #Save to Excel  
    excel_file_path = f'{output_dir}\\results.xlsx'
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_df.to_excel(writer, index=False, sheet_name=f'{algo_column}')

    return fpr, tpr, roc_auc, accuracy, precision, recall, f1, cf_matrix

def compute_metrics_subgroups(X, y, data_dicts, subgroup_key, subgroup_value, algo_column, output_dir):
    """Function to compute metrics for specific subgroups

    Args:
        X: Predictions
        y: Ground truth labels
        data_dicts: List of dictionaries containing the data
        subgroup_key: Key to filter the subgroup (e.g., 'brand', 'sex')
        subgroup_value: Value of the subgroup to filter (e.g., 'endA', 'Female')

    Returns:
        fpr: False Positive Rate    
        tpr: True Positive Rate
        roc_auc: Area Under the Curve
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
        cf_matrix: Confusion matrix
    """

    X_subgroup = [] # Predictions for the subgroup
    y_subgroup = [] # Ground truth labels for the subgroup

    for i, entry in enumerate(data_dicts):
        if entry[subgroup_key] == subgroup_value:
            X_subgroup.append(X[i])
            y_subgroup.append(y[i])

    X_subgroup = np.array(X_subgroup)
    y_subgroup = np.array(y_subgroup)

    # Compute metrics for the specific subgroup
    fpr, tpr, roc_auc, accuracy, precision, recall, f1, cf_matrix = compute_metrics(X_subgroup, y_subgroup, algo_column=f"{algo_column}_{subgroup_key}_{subgroup_value}", output_dir=output_dir)

    return fpr, tpr, roc_auc, accuracy, precision, recall, f1, cf_matrix 


def save_confusion_matrix_as_image(cf_matrix, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('GT')
    plt.savefig(filename)
    plt.close()


def save_combined_predictions(csv_file_path, data_dicts):
    """Function to save combined predictions from both software team and AI team to a unique CSV file

    Args:
        csv_file_path: Path to the original "cpp_output.csv" file
        data_dicts: List of dictionaries containing the data

    Returns:
        None
    """

    output_file = csv_file_path.replace("cpp_output.csv", "combined_cpp_predictions.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["frameID", "patientID", "age", "sex", "brand", "GT", "algo1", "algo2", "illumination"])

        # Write the data
        for entry in data_dicts:
            writer.writerow([entry[key] for key in ["frameID", "patientID", "age", "sex", "brand", "GT", "algo1", "algo2", "illumination"]])

    print(f"Combined predictions saved to {output_file}")

def calculate_metrics_and_stats_by_subgroup(csv_file_path, subgroup_column, gt_column, algo_column, group1, group2, output_dir):
    """Calculate metrics for specific subgroups and perform statistical tests.

    Args:
        csv_file_path (str): Path to the CSV file.
        subgroup_column (str): Column to filter subgroups (e.g., 'sex', 'brand').
        gt_column (str): Column containing ground truth labels (e.g., 'GT').
        algo_column (str): Column containing algorithm predictions (e.g., 'algo1' or 'algo2').
        group1 (str): First subgroup value (e.g., 'Male').
        group2 (str): Second subgroup value (e.g., 'Female').

    Returns:
        None
    """
    # Load the data
    df = pd.read_csv(csv_file_path)

    # Filter data by subgroups
    group1_data = df[df[subgroup_column] == group1]
    group2_data = df[df[subgroup_column] == group2]

    # Extract ground truth and predictions for each group
    y_true_group1 = [1 if x == 'P' else 0 for x in group1_data[gt_column]]
    y_pred_group1 = [1 if x == 'P' else 0 for x in group1_data[algo_column]]

    y_true_group2 = [1 if x == 'P' else 0 for x in group2_data[gt_column]]
    y_pred_group2 = [1 if x == 'P' else 0 for x in group2_data[algo_column]]

    # Convert pandas Series to numpy arrays for element-wise multiplication
    y_true_group1 = np.array(y_true_group1)
    y_pred_group1 = np.array(y_pred_group1)
    y_true_group2 = np.array(y_true_group2)
    y_pred_group2 = np.array(y_pred_group2)

    # Calculate metrics for each group
    precision1 = precision_score(y_true_group1, y_pred_group1, zero_division=0)
    recall1 = recall_score(y_true_group1, y_pred_group1, zero_division=0)
    f1_1 = f1_score(y_true_group1, y_pred_group1, zero_division=0)

    precision2 = precision_score(y_true_group2, y_pred_group2, zero_division=0)
    recall2 = recall_score(y_true_group2, y_pred_group2, zero_division=0)
    f1_2 = f1_score(y_true_group2, y_pred_group2, zero_division=0)

    print(f"Metrics for {group1}:")
    print(f"Precision: {precision1:.2f}, Recall: {recall1:.2f}, F1 Score: {f1_1:.2f}")

    print(f"Metrics for {group2}:")
    print(f"Precision: {precision2:.2f}, Recall: {recall2:.2f}, F1 Score: {f1_2:.2f}")

    # Perform element-wise multiplication
    f1_scores_group1 = y_true_group1 * y_pred_group1  # True positives for group1
    f1_scores_group2 = y_true_group2 * y_pred_group2  # True positives for group2
    print("F1 Scores Group 1:", f1_scores_group1)

    # Perform Mann-Whitney U test on F1 scores
    stat, p_value = mannwhitneyu(f1_scores_group1, f1_scores_group2, alternative='two-sided')
    print(f"Mann-Whitney U Test: U={stat}, p-value={p_value}")

    # Save confusion matrices and metrics side by side in a single plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot confusion matrix for group 1
    sns.heatmap(confusion_matrix(y_true_group1, y_pred_group1), annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix: {group1}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Plot confusion matrix for group 2
    sns.heatmap(confusion_matrix(y_true_group2, y_pred_group2), annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[1])
    axes[1].set_title(f'Confusion Matrix: {group2}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    # Plot metrics and p-value
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        group1: [precision1, recall1, f1_1],
        group2: [precision2, recall2, f1_2]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.plot(kind='bar', x='Metric', ax=axes[2], color=['blue', 'orange'], legend=True)
    axes[2].set_title('Metrics Comparison')
    axes[2].set_ylabel('Score')

    # Add p-value annotation
    if p_value < 0.05:
        axes[2].text(1, max(max(metrics_data[group1]), max(metrics_data[group2])), '*: p-value<0.05', 
                     ha='center', va='bottom', color='red', fontsize=12)
    else:
        axes[2].text(1, max(max(metrics_data[group1]), max(metrics_data[group2])), 'p-value>=0.05', 
                     ha='center', va='bottom', color='black', fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}\\Figs\\{algo_column}_metrics_and_confusion_matrices_{group1}_vs_{group2}.png')
    plt.show()

    # Save metrics to an Excel file
    results_data = {
        'Algorithm': [algo_column, algo_column],
        'Group': [group1, group2],
        'Precision': [precision1, precision2],
        'Recall': [recall1, recall2],
        'F1 Score': [f1_1, f1_2],
        'P-Value': [p_value, p_value],
        'Significance': ['*' if p_value < 0.05 else ''] * 2
    }
    results_df = pd.DataFrame(results_data)

    # Save to Excel
    excel_file_path = f'{output_dir}\\results.xlsx'
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_df.to_excel(writer, index=False, sheet_name=f'{group1}_vs_{group2}')

    print(f"Metrics saved to Excel file: {excel_file_path}")

def compare_subgroups_with_different_algorithm(csv_file_path, subgroup_column, gt_column, algo_column_1, algo_column_2, group, output_dir):
    """Compare two subgroups using the same algorithm and calculate metrics.

    Args:
        csv_file_path (str): Path to the CSV file.
        subgroup_column (str): Column to filter subgroups (e.g., 'sex', 'brand').
        gt_column (str): Column containing ground truth labels (e.g., 'GT').
        algo_column_1 (str): Column containing first algorithm predictions (e.g., 'algo1').
        algo_column_2 (str): Column containing second algorithm predictions (e.g., 'algo2').
        group (str): Subgroup value (e.g., 'Male').
    .output_dir (str): Path to save the output files.

    Returns:
        None
    """
    # Load the data
    df = pd.read_csv(csv_file_path)

    # Filter data by subgroups
    subgroup_data = df[df[subgroup_column] == group]

    # Extract ground truth and predictions for each group
    y_true_group = [1 if x == 'P' else 0 for x in subgroup_data[gt_column]]
    y_pred_group1 = [1 if x == 'P' else 0 for x in subgroup_data[algo_column_1]]
    y_pred_group2 = [1 if x == 'P' else 0 for x in subgroup_data[algo_column_2]]

    # Convert pandas Series to numpy arrays for element-wise multiplication
    y_true_group = np.array(y_true_group)
    y_pred_group1 = np.array(y_pred_group1)
    y_pred_group2 = np.array(y_pred_group2)

    # Calculate metrics for each group
    precision1 = precision_score(y_true_group, y_pred_group1, zero_division=0)
    recall1 = recall_score(y_true_group, y_pred_group1, zero_division=0)
    f1_1 = f1_score(y_true_group, y_pred_group1, zero_division=0)

    precision2 = precision_score(y_true_group, y_pred_group2, zero_division=0)
    recall2 = recall_score(y_true_group, y_pred_group2, zero_division=0)
    f1_2 = f1_score(y_true_group, y_pred_group2, zero_division=0)

    print(f"Metrics for {algo_column_1}:")
    print(f"Precision: {precision1:.2f}, Recall: {recall1:.2f}, F1 Score: {f1_1:.2f}")

    print(f"Metrics for {algo_column_2}:")
    print(f"Precision: {precision2:.2f}, Recall: {recall2:.2f}, F1 Score: {f1_2:.2f}")

    # Perform element-wise multiplication
    f1_scores_group1 = y_true_group * y_pred_group1  # True positives for group1
    f1_scores_group2 = y_true_group * y_pred_group2  # True positives for group2
    print("F1 Scores Group 1:", f1_scores_group1)

    # Perform Mann-Whitney U test on F1 scores
    stat, p_value = mannwhitneyu(f1_scores_group1, f1_scores_group2, alternative='two-sided')
    print(f"Mann-Whitney U Test: U={stat}, p-value={p_value}")

    # Save confusion matrices and metrics side by side in a single plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot confusion matrix for group 1
    sns.heatmap(confusion_matrix(y_true_group, y_pred_group1), annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix: {algo_column_1}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Plot confusion matrix for group 2
    sns.heatmap(confusion_matrix(y_true_group, y_pred_group2), annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[1])
    axes[1].set_title(f'Confusion Matrix: {algo_column_2}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    # Plot metrics and p-value
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        algo_column_1: [precision1, recall1, f1_1],
        algo_column_2: [precision2, recall2, f1_2]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.plot(kind='bar', x='Metric', ax=axes[2], color=['blue', 'orange'], legend=True)
    axes[2].set_title('Metrics Comparison')
    axes[2].set_ylabel('Score')

    # Add p-value annotation
    if p_value < 0.05:
        axes[2].text(1, max(max(metrics_data[algo_column_1]), max(metrics_data[algo_column_2])), '*: p-value<0.05', 
                     ha='center', va='bottom', color='red', fontsize=12)
    else:
        axes[2].text(1, max(max(metrics_data[algo_column_1]), max(metrics_data[algo_column_2])), 'p-value>=0.05', 
                     ha='center', va='bottom', color='black', fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}\\Figs\\{group}_metrics_and_confusion_matrices_{algo_column_1}_vs_{algo_column_2}.png')
    plt.show()

    results_data = {
        'Algorithm': [algo_column_1, algo_column_2],
        'Group': [group, group],
        'Precision': [precision1, precision2],
        'Recall': [recall1, recall2],
        'F1 Score': [f1_1, f1_2],
        'P-Value': [p_value, p_value],
        'Significance': ['*' if p_value < 0.05 else ''] * 2
    }
    results_df = pd.DataFrame(results_data)

    # Save to Excel
    excel_file_path = f'{output_dir}\\results.xlsx'
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_df.to_excel(writer, index=False, sheet_name=f'{group}_{algo_column_1}_vs_{algo_column_2}')

    print(f"Metrics saved to Excel file: {excel_file_path}")

def plot_roc_curve(fpr, tpr, auc_score, algo_name, output_path):
    """Plot ROC curve and calculate AUC.

    Args:
        fpr (array): False Positive Rate.
        tpr (array): True Positive Rate.
        auc_score (float): Area Under the Curve (AUC) score.
        algo_name (str): Name of the algorithm.
        output_path (str): Path to save the ROC curve plot.

    Returns:
        auc_score (float): Area Under the Curve (AUC) score.
    """

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {algo_name}')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    plot_path = f"{output_path}\\Figs\\{algo_name}_roc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve for {algo_name} saved to {plot_path}")
    return auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cpp_output.csv and compute metrics.")
    parser.add_argument("--csv_file_path", type=str, required=True, help="Path to the cpp_output.csv file.") #'.\\Baldini_Detection_Assignment\\cpp_output.csv'
    parser.add_argument("--subgroup_column", type=str, help="Column to filter subgroups (e.g., 'sex', 'brand').")
    parser.add_argument("--gt_column", type=str, help="Column containing ground truth labels (e.g., 'GT').")
    parser.add_argument("--algo_column", type=str, help="Column containing algorithm predictions (e.g., 'algo1' or 'algo2').")
    parser.add_argument("--algo_column2", type=str, help="Column containing algorithm predictions (e.g., 'algo1' or 'algo2').")
    parser.add_argument("--group1", type=str, help="First subgroup value (e.g., 'Male').")
    parser.add_argument("--group2", type=str, help="Second subgroup value (e.g., 'Female').")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output files.") #'.\\Baldini_Detection_Assignment'
    args = parser.parse_args()

    # Load and process data
    X_algo1, y, X_algo2 = process_cpp_outputs(args.csv_file_path)

    # Process data_dicts again to save combined predictions
    data = np.loadtxt(args.csv_file_path, dtype=str, delimiter=',', skiprows=1)
    keys = ["frameID", "patientID", "age", "sex", "brand", "GT", "algo1", "algo2", "illumination"]
    data_dicts = [dict(zip(keys, row)) for row in data]
    print("Totlal entries processed = total number of frames:", len(data_dicts))
    save_combined_predictions(args.csv_file_path, data_dicts)

    # print("Algorithm 1 Performance:")
    # fpr1, tpr1, roc_auc1, accuracy1, precision1, recall1, f1_1, cf_matrix1 = compute_metrics(X_algo1, y, algo_column='algo1',output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix1)
    # print(f"Accuracy: {accuracy1:.2f}")
    # print(f"Precision: {precision1:.2f}")
    # print(f"Recall: {recall1:.2f}")
    # print(f"F1 Score: {f1_1:.2f}")
    # print(f"AUC: {roc_auc1:.2f}")
    # plot_roc_curve(fpr1, tpr1, roc_auc1, 'algo1', args.output_dir)
    # # save_confusion_matrix_as_image(cf_matrix1, f'{args.output_dir}\\confusion_matrix_algo1.png')

    # print("\nAlgorithm 2 Performance:")
    # fpr2, tpr2, roc_auc2, accuracy2, precision2, recall2, f1_2, cf_matrix2 = compute_metrics(X_algo2, y, algo_column='algo2',output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2)
    # print(f"Accuracy: {accuracy2:.2f}")
    # print(f"Precision: {precision2:.2f}")
    # print(f"Recall: {recall2:.2f}")
    # print(f"F1 Score: {f1_2:.2f}")
    # print(f"AUC: {roc_auc2:.2f}")
    # plot_roc_curve(fpr2, tpr2, roc_auc2, 'algo2', args.output_dir)
    # save_confusion_matrix_as_image(cf_matrix2, f'{args.output_dir}\\confusion_matrix_algo2.png')

    # # Example of computing metrics for a subgroup (e.g. brand 'BrandA')
    # sex_value = 'Female'
    # print(f"\nAlgorithm 1 Performance for sex = {sex_value}:")
    # fpr1_s, tpr1_s, roc_auc1_s, accuracy1_s, precision1_s, recall1_s, f1_1_s, cf_matrix1_s = compute_metrics_subgroups(X_algo1, y, data_dicts, 'sex', sex_value, algo_column="algo1",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix1_s)   
    # print(f"Accuracy: {accuracy1_s:.2f}")
    # print(f"Precision: {precision1_s:.2f}")
    # print(f"Recall: {recall1_s:.2f}")
    # print(f"F1 Score: {f1_1_s:.2f}")
    # save_confusion_matrix_as_image(cf_matrix1_s, f'{args.output_dir}\\confusion_matrix_algo1_sex_{sex_value}.png')

    # sex_value = 'Male'
    # print(f"\nAlgorithm 1 Performance for sex = {sex_value}:")
    # fpr1_s, tpr1_s, roc_auc1_s, accuracy1_s, precision1_s, recall1_s, f1_1_s, cf_matrix1_s = compute_metrics_subgroups(X_algo1, y, data_dicts, 'sex', sex_value, algo_column="algo1",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix1_s)   
    # print(f"Accuracy: {accuracy1_s:.2f}")
    # print(f"Precision: {precision1_s:.2f}")
    # print(f"Recall: {recall1_s:.2f}")
    # print(f"F1 Score: {f1_1_s:.2f}")
    # save_confusion_matrix_as_image(cf_matrix1_s, f'{args.output_dir}\\confusion_matrix_algo1_sex_{sex_value}.png')

    # brand_value = 'endA' 
    # print(f"\nAlgorithm 1 Performance for brand = {brand_value}:")
    # fpr2_b, tpr2_b, roc_auc2_b, accuracy2_b, precision2_b, recall2_b, f1_2_b, cf_matrix2_b = compute_metrics_subgroups(X_algo1, y, data_dicts, 'brand', brand_value, algo_column="algo1",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2_b)        
    # print(f"Accuracy: {accuracy2_b:.2f}")
    # print(f"Precision: {precision2_b:.2f}")     
    # print(f"Recall: {recall2_b:.2f}")
    # print(f"F1 Score: {f1_2_b:.2f}")
    # save_confusion_matrix_as_image(cf_matrix2_b, f'{args.output_dir}\\confusion_matrix_algo1_brand_{brand_value}.png')   

    # brand_value = 'endA' 
    # print(f"\nAlgorithm 2 Performance for brand = {brand_value}:")
    # fpr2_b, tpr2_b, roc_auc2_b, accuracy2_b, precision2_b, recall2_b, f1_2_b, cf_matrix2_b = compute_metrics_subgroups(X_algo2, y, data_dicts, 'brand', brand_value, algo_column="algo2",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2_b)        
    # print(f"Accuracy: {accuracy2_b:.2f}")
    # print(f"Precision: {precision2_b:.2f}")     
    # print(f"Recall: {recall2_b:.2f}")
    # print(f"F1 Score: {f1_2_b:.2f}")
    # save_confusion_matrix_as_image(cf_matrix2_b, f'{args.output_dir}\\confusion_matrix_algo2_brand_{brand_value}.png')   
    
    # brand_value = 'endB' 
    # print(f"\nAlgorithm 2 Performance for brand = {brand_value}:")  
    # fpr2_b, tpr2_b, roc_auc2_b, accuracy2_b, precision2_b, recall2_b, f1_2_b, cf_matrix2_b = compute_metrics_subgroups(X_algo2, y, data_dicts, 'brand', brand_value, algo_column="algo2",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2_b)    
    # print(f"Accuracy: {accuracy2_b:.2f}")
    # print(f"Precision: {precision2_b:.2f}")
    # print(f"Recall: {recall2_b:.2f}")
    # print(f"F1 Score: {f1_2_b:.2f}")
    # save_confusion_matrix_as_image(cf_matrix2_b, f'{args.output_dir}\\confusion_matrix_algo2_brand_{brand_value}.png')       

    # brand_value = 'endC' 
    # print(f"\nAlgorithm 2 Performance for brand = {brand_value}:")  
    # fpr2_b, tpr2_b, roc_auc2_b, accuracy2_b, precision2_b, recall2_b, f1_2_b, cf_matrix2_b = compute_metrics_subgroups(X_algo2, y, data_dicts, 'brand', brand_value, algo_column="algo2",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2_b)    
    # print(f"Accuracy: {accuracy2_b:.2f}")
    # print(f"Precision: {precision2_b:.2f}")
    # print(f"Recall: {recall2_b:.2f}")
    # print(f"F1 Score: {f1_2_b:.2f}")
    # save_confusion_matrix_as_image(cf_matrix2_b, f'{args.output_dir}\\confusion_matrix_algo2_brand_{brand_value}.png')

    # brand_value = 'endD' 
    # print(f"\nAlgorithm 2 Performance for brand = {brand_value}:")  
    # fpr2_b, tpr2_b, roc_auc2_b, accuracy2_b, precision2_b, recall2_b, f1_2_b, cf_matrix2_b = compute_metrics_subgroups(X_algo2, y, data_dicts, 'brand', brand_value, algo_column="algo2",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2_b)    
    # print(f"Accuracy: {accuracy2_b:.2f}")
    # print(f"Precision: {precision2_b:.2f}")
    # print(f"Recall: {recall2_b:.2f}")
    # print(f"F1 Score: {f1_2_b:.2f}")
    # save_confusion_matrix_as_image(cf_matrix2_b, f'{args.output_dir}\\confusion_matrix_algo2_brand_{brand_value}.png')

    # ill_value = 'Low' 
    # print(f"\nAlgorithm 1 Performance for ill = {ill_value}:")  
    # fpr2_b, tpr2_b, roc_auc2_b, accuracy2_b, precision2_b, recall2_b, f1_2_b, cf_matrix2_b = compute_metrics_subgroups(X_algo1, y, data_dicts, 'illumination', ill_value, algo_column="algo1",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2_b)    
    # print(f"Accuracy: {accuracy2_b:.2f}")
    # print(f"Precision: {precision2_b:.2f}")
    # print(f"Recall: {recall2_b:.2f}")
    # print(f"F1 Score: {f1_2_b:.2f}")
    # save_confusion_matrix_as_image(cf_matrix2_b, f'{args.output_dir}\\confusion_matrix_algo1_illumination_{ill_value}.png')

    # ill_value = 'High' 
    # print(f"\nAlgorithm 1 Performance for ill = {ill_value}:")  
    # fpr2_b, tpr2_b, roc_auc2_b, accuracy2_b, precision2_b, recall2_b, f1_2_b, cf_matrix2_b = compute_metrics_subgroups(X_algo1, y, data_dicts, 'illumination', ill_value, algo_column="algo1",output_dir=f'{args.output_dir}')
    # print("\nConfusion Matrix:\n", cf_matrix2_b)    
    # print(f"Accuracy: {accuracy2_b:.2f}")
    # print(f"Precision: {precision2_b:.2f}")
    # print(f"Recall: {recall2_b:.2f}")
    # print(f"F1 Score: {f1_2_b:.2f}")
    # save_confusion_matrix_as_image(cf_matrix2_b, f'{args.output_dir}\\confusion_matrix_algo1_illumination_{ill_value}.png')

    # Example of calculating metrics for specific subgroups
    # calculate_metrics_and_stats_by_subgroup(
    #     csv_file_path=args.csv_file_path,
    #     subgroup_column=args.subgroup_column,
    #     gt_column=args.gt_column,
    #     algo_column=args.algo_column,
    #     group1=args.group1,
    #     group2=args.group2,
    #     output_dir=args.output_dir
    # )

    compare_subgroups_with_different_algorithm(
        csv_file_path=args.csv_file_path,
        subgroup_column=args.subgroup_column,
        gt_column=args.gt_column,
        algo_column_1=args.algo_column,
        algo_column_2=args.algo_column2,
        group=args.group1,
        output_dir=args.output_dir
    )

    # TO RUN: 
    # >> python .\process_cpp_outputs_and_compare.py --csv_file_path "C:\Users\cbaldini\Desktop\Baldini_Detection_Assignment\Baldini_Detection_Assignment\cpp_output.csv" --subgroup_column "brand" --gt_column "GT" --algo_column "algo2" --group1 "endA" --group2 "endC" -.output_dir "C:\Users\cbaldini\Desktop\Baldini_Detection_Assignment\Baldini_Detection_Assignment"
    # >> python .\process_cpp_outputs_and_compare.py --csv_file_path "C:\Users\cbaldini\Desktop\Baldini_Detection_Assignment\Baldini_Detection_Assignment\cpp_output.csv" --subgroup_column "brand" --gt_column "GT" --algo_column "algo2" --group1 "endD" --group2 "endC" --output_path "C:\Users\cbaldini\Desktop\Baldini_Detection_Assignment\Baldini_Detection_Assignment" --algo_column2 "algo1"                

