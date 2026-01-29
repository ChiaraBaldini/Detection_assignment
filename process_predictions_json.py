import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import itertools
import argparse

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def process_json_data(data):
    """Process the JSON data to create a comparison dictionary."""
    comparison_dict = {}

    for entry in data:
        frame_patient_id = entry["frameID_patientID"]
        frameID=frame_patient_id.split("_")[0]
        patientID=frame_patient_id.split("_")[1]
        gt_bboxes = entry.get("gt", {}).get("bboxes", [])
        algo1_bboxes = entry.get("algo1", [])
        algo2_bboxes = entry.get("algo2", [])
        
        # Merge the information into the main dictionary
        comparison_dict[frame_patient_id] = {
            "frameID": frameID,
            "patientID": patientID,
            "ground_truth": gt_bboxes,
            "algo1": algo1_bboxes,
            "algo2": algo2_bboxes
        }

    return comparison_dict

def calculate_iou(box1, box2, height=512, width=512):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.
    Each box is represented as [x_center, y_center, width, height, confidence]
    Args:   
        box1: List or array of 5 elements [x_center, y_center, width, height, confidence]
        box2: List or array of 5 elements [x_center, y_center, width, height, confidence]
        height: Height of the image (default: 512)
        width: Width of the image (default: 512)
    Returns:
        iou: Intersection over Union (IoU) value between box1 and box2
    """
    x1 = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
    y1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    x2 = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
    y2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)

    # Compute the area of intersection rectangle
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # Compute the IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def update_combined_predictions(json_data, combined_csv_path):
    """Update combined_predictions.csv with JSON data.
    Args:
        json_data: List of dictionaries containing JSON data
        combined_csv_path: Path to the combined_predictions.csv file
    Returns:
        None
    """
    # Read the existing combined_predictions.csv
    with open(combined_csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        combined_data = list(reader)

    # Create an index for faster lookup
    combined_data_index = {f"{row['frameID']}_{row['patientID']}": row for row in combined_data if 'frameID' in row and 'patientID' in row}

    # Check and update rows

    for entry in json_data:
        # print("Processing entry for frameID_patientID:", entry["frameID_patientID"])
        check=False
        frame_patient_id = entry["frameID_patientID"]
        frameID = frame_patient_id.split("_")[0]
        patientID = frame_patient_id.split("_")[1]
        gt_bboxes = entry.get("gt", {}).get("bboxes", [])
        algo1_bboxes = entry.get("algo1", [])
        algo2_bboxes = entry.get("algo2", [])
        # print("DEBUG:", frame_patient_id, gt_bboxes, algo1_bboxes, algo2_bboxes)

        # Find the corresponding row in combined_predictions.csv using the index
        matching_row = combined_data_index.get(frame_patient_id)

        if matching_row:
            # Update the row with JSON data
            matching_row["ground_truth_bb"] = json.dumps(gt_bboxes)
            matching_row["algo1_bb"] = json.dumps(algo1_bboxes)
            matching_row["algo2_bb"] = json.dumps(algo2_bboxes)

            # # Check for inconsistencies
            # if not gt_bboxes and matching_row["GT"] == "P":
            #     print(f"Inconsistency found for frameID_patientID: {frame_patient_id} - GT is 'P' but ground_truth is empty.")

            # # Check IoU condition for algo1
            # if matching_row["algo1"] == "P":
            #     check = False
            #     for gt_box in gt_bboxes:
            #         for algo1_box in algo1_bboxes:
            #             iou = calculate_iou(gt_box, algo1_box)
            #             # print("DEBUG: IoU =", iou, "with confidence:", algo1_box[4])
            #             if iou > 0.2 and algo1_box[4] > 0.7:
            #                 check = True  # Condition satisfied
            #                 break  # Exit inner loop
            #         if check:
            #             # print(f"DEBUG: Condition satisfied for frameID_patientID: {frame_patient_id} with IoU: {iou} and confidence: {algo1_box[4]}")
            #             break  # Exit outer loop
            #     if not check:
            #         print(f"Inconsistency found for frameID_patientID: {frame_patient_id} - algo1 is 'P' but no IoU > 0.7 between ground_truth_bb and algo1_bb.")
        else:
            # Print missing rows
            print(f"Missing row for frameID_patientID: {frame_patient_id} in combined_predictions.csv")

    # Update fieldnames to include new columns if not already present
    fieldnames = reader.fieldnames
    new_fields = ["ground_truth_bb", "algo1_bb", "algo2_bb"]
    for field in new_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    # Write the updated data back to combined_predictions.csv
    with open(combined_csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_data)

def calculate_all_ious(gt_bboxes, algo_bboxes):
    """Calculate IoUs for all pairs of ground truth and algorithm bounding boxes, excluding IoU = 0.0."""
    ious = []
    for gt_box in gt_bboxes:
        for algo_box in algo_bboxes:
            iou = calculate_iou(gt_box, algo_box)
            if iou > 0.0:  # Exclude IoU = 0.0
                ious.append(iou)
    return ious

def parse_bounding_boxes(bboxes_str):
    """Parse bounding boxes from a JSON string to a list of lists of floats."""
    try:
        bboxes = json.loads(bboxes_str)
        return [list(map(float, bbox)) for bbox in bboxes]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing bounding boxes: {bboxes_str}. Error: {e}")
        return []
    
def save_confusion_matrix_as_image(cf_matrix, filename):
    """Save confusion matrix as an image file."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('GT')
    plt.savefig(filename)
    plt.close()

def generate_boxplot_and_test_from_csv(combined_csv_path, output_dir):
    """Generate boxplots for IoUs of algo1 and algo2 from combined_predictions.csv, perform a statistical test, and save the plot.
    Args:
        combined_csv_path: Path to the combined_predictions.csv file
        output_dir: Directory to save the output plot
    Returns:    
        None
    """
    algo1_ious = []
    algo2_ious = []

    # Read the combined_predictions.csv
    with open(combined_csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Consider only rows where GT is 'P'
            if row["GT"] != "P":
                continue

            gt_bboxes = parse_bounding_boxes(row["ground_truth_bb"])
            algo1_bboxes = parse_bounding_boxes(row["algo1_bb"])
            algo2_bboxes = parse_bounding_boxes(row["algo2_bb"])

            # Calculate IoUs for algo1 and algo2
            algo1_ious.extend(calculate_all_ious(gt_bboxes, algo1_bboxes))
            algo2_ious.extend(calculate_all_ious(gt_bboxes, algo2_bboxes))

    print(f"Total IoUs calculated - Algorithm 1: {len(algo1_ious)}, Algorithm 2: {len(algo2_ious)}")
    print(f"Sample IoUs - Algorithm 1: {algo1_ious[:5]}, Algorithm 2: {algo2_ious[:5]}")

    # Check if IoU lists are non-empty
    if not algo1_ious or not algo2_ious:
        print("Error: One or both IoU lists are empty. Ensure there are entries with GT = 'P' in the data.")
        return

    # Perform statistical test (Mann-Whitney U test)
    stat, p_value = mannwhitneyu(algo1_ious, algo2_ious, alternative='two-sided')
    print(f"Mann-Whitney U Test: U={stat}, p-value={p_value}")

    # Calculate medians
    median_algo1 = np.median(algo1_ious)
    median_algo2 = np.median(algo2_ious)
    print(f"Median IoU for Algorithm 1: {median_algo1}")
    print(f"Median IoU for Algorithm 2: {median_algo2}")

    # Generate boxplot
    data = [algo1_ious, algo2_ious]
    labels = ['Algorithm 1', 'Algorithm 2']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, palette=['blue', 'orange'])
    plt.xticks(ticks=[0, 1], labels=labels)
    plt.title('IoU Comparison Between Algorithm 1 and Algorithm 2 (GT = P)')
    plt.ylabel('IoU')

    # Add median values as text on the plot
    for i, median in enumerate([median_algo1, median_algo2]):
        plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    # Add statistical significance annotation
    if p_value < 0.05:
        max_value = max(max(algo1_ious), max(algo2_ious))
        plt.text(0.5, max_value - 0.05, '*: p-value<0.05', ha='center', va='bottom', color='red', fontsize=12)
        print("Statistically significant difference between Algorithm 1 and Algorithm 2 IoUs (p < 0.05).")
    else:
        print("No statistically significant difference between Algorithm 1 and Algorithm 2 IoUs (p >= 0.05).")

    # Save the plot
    plt.savefig(f'{output_dir}\\Figs\\iou_comparison_boxplot_with_stats_and_medians_gt_p_from_csv.png')
    plt.show()


def generate_scatter_plot_iou_confidence_with_colored_edges(combined_csv_path, output_dir):
    """Generate a scatter plot combining IoU and confidence for algo1 and algo2, with point colors based on algorithm and edge colors based on labels.
    For each ground truth bounding box, only the predicted bounding box with the highest IoU is considered for algo1 and algo2.
    """
    algo1_data = []
    algo2_data = []

    # Read the combined_predictions.csv
    with open(combined_csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["GT"] == "P":
                algo1_label = row["algo1"]  # Get the algo1 label (P or N)
                algo2_label = row["algo2"]  # Get the algo2 label (P or N)

                gt_bboxes = parse_bounding_boxes(row["ground_truth_bb"])
                algo1_bboxes = parse_bounding_boxes(row["algo1_bb"])
                algo2_bboxes = parse_bounding_boxes(row["algo2_bb"])

                # Collect IoU, confidence, and label for algo1
                for gt_box in gt_bboxes:
                    best_iou = 0
                    best_conf = 0
                    best_algo1_label = None
                    for algo1_box in algo1_bboxes:
                        iou = calculate_iou(gt_box, algo1_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_conf = algo1_box[4]
                            # Original label assignment for algo1
                            best_algo1_label = algo1_label

                            # # Fixed label assignment for algo1
                            # if iou>0.2 and algo1_box[4]>=0.7:
                            #     best_algo1_label = "P"
                            # else:
                            #     best_algo1_label = "N"

                    if best_iou > 0:  # Only add if a match was found
                        algo1_data.append((best_iou, best_conf, best_algo1_label))

                # Collect IoU, confidence, and label for algo2
                for gt_box in gt_bboxes:
                    best_iou = 0
                    best_conf = 0
                    best_algo2_label = None
                    for algo2_box in algo2_bboxes:
                        iou = calculate_iou(gt_box, algo2_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_conf = algo2_box[4]
                            # Original label assignment for algo2
                            best_algo2_label = algo2_label

                            # # Fixed label assignment for algo2
                            # if iou>0.2 and algo2_box[4]>=0.5:
                            #     best_algo2_label = "P"
                            # else:
                            #     best_algo2_label = "N"

                    if best_iou > 0:  # Only add if a match was found
                        algo2_data.append((best_iou, best_conf, best_algo2_label))

    # Separate IoU, confidence, and labels for plotting
    algo1_ious, algo1_confidences, algo1_labels = zip(*algo1_data) if algo1_data else ([], [], [])
    algo2_ious, algo2_confidences, algo2_labels = zip(*algo2_data) if algo2_data else ([], [], [])

    # Map labels to edge colors (P -> green, N -> red)
    label_to_edgecolor = {"P": "green", "N": "red"}
    algo1_edgecolors = [label_to_edgecolor[label] for label in algo1_labels]
    algo2_edgecolors = [label_to_edgecolor[label] for label in algo2_labels]

    # Generate enhanced scatter plot with color-coded edges
    plt.figure(figsize=(12, 8))
    scatter1 = plt.scatter(algo1_confidences, algo1_ious, label='Algorithm 1', alpha=0.6, color='blue', edgecolor=algo1_edgecolors, s=20, marker='o') #was c=algo1_ious, cmap='Blues', s=np.array(algo1_confidences) * 100
    scatter2 = plt.scatter(algo2_confidences, algo2_ious, label='Algorithm 2', alpha=0.6, color='yellow', edgecolor=algo2_edgecolors, s=20, marker='s')

    # # Add colorbars for the gradients
    # cbar1 = plt.colorbar(scatter1, label='IoU (Algorithm 1)', pad=0.01)
    # cbar2 = plt.colorbar(scatter2, label='IoU (Algorithm 2)', pad=0.01)

    # Add threshold lines
    plt.axvline(x=0.5, color='orange', linestyle='--', label='Algorithm 2 Confidence Threshold (0.5)')       
    plt.axhline(y=0.2, color='black', linestyle='--', label='IoU Threshold (0.2)')
    plt.axvline(x=0.7, color='blue', linestyle='--', label='Algorithm 1 Confidence Threshold (0.7)') 

    # Add trendlines
    if algo1_data:
        z1 = np.polyfit(algo1_confidences, algo1_ious, 1)
        p1 = np.poly1d(z1)
        plt.plot(algo1_confidences, p1(algo1_confidences), color='blue', label='Trendline Algorithm 1')
    if algo2_data:
        z2 = np.polyfit(algo2_confidences, algo2_ious, 1)
        p2 = np.poly1d(z2)
        plt.plot(algo2_confidences, p2(algo2_confidences), color='orange', label='Trendline Algorithm 2')
    # Add labels and title
    plt.title('IoU vs Confidence for Algorithm 1 and Algorithm 2 (with colored edges based on P or N)', fontsize=16)
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('IoU', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(f'{output_dir}\\Figs\\iou_vs_confidence_scatter_plot_with_colored_edges.png', dpi=300, bbox_inches='tight')
    plt.show()

# Function to generate Excel file from CSV
def generate_excel_from_csv(combined_csv_path, output_excel_path):
    """Generate an Excel file from combined_predictions.csv with all data."""
    # Read the combined_predictions.csv
    with open(combined_csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save to Excel
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"Excel file saved to {output_excel_path}")

# Function to generate Excel file with additional metrics
def generate_excel_with_metrics(combined_csv_path, output_excel_path):
    """Generate an Excel file from combined_predictions.csv with additional metrics (max IoU and confidence for algo1 and algo2)."""
    # Read the combined_predictions.csv
    reader = pd.read_excel(output_excel_path)
    data = []

    for _, row in reader.iterrows():
        algo1_label = row["algo1"]
        algo2_label = row["algo2"]
        gt_label= row["GT"]
        
        gt_bboxes = parse_bounding_boxes(row["ground_truth_bb"])
        algo1_bboxes = parse_bounding_boxes(row["algo1_bb"])
        algo2_bboxes = parse_bounding_boxes(row["algo2_bb"])

        iou_values1 = []
        conf_values1 = []
        original_labels1 = []
        reviewed_labels1 = []
        iou_values2 = []
        conf_values2 = []
        original_labels2 = []
        reviewed_labels2 = []

        if gt_bboxes==[] and gt_label == "P":
            row["iou_algo1"] = [0]
            row["conf_algo1"] = [0]
            row["iou_algo2"] = [0]
            row["conf_algo2"] = [0]
            row["original_label_gt"] =gt_label
            row["reviewed_label_gt"] = ["N"] 
            row["note"] = "Inconsistency: GT is 'P' but ground_truth is empty."
        
        else:
            row["original_label_gt"] =gt_label
            row["reviewed_label_gt"] = gt_label
            # Process IoU and confidence for algo1

            for gt_box in gt_bboxes:
                best_iou1 = 0
                best_conf1 = 0
                best_label1 = "N"
                for algo1_box in algo1_bboxes:
                    iou = calculate_iou(gt_box, algo1_box)
                    if iou > best_iou1:
                        best_iou1 = iou
                        best_conf1 = algo1_box[4]
                        # Original label assignment for algo1
                        best_label1 = algo1_label

                iou_values1.append(best_iou1)
                conf_values1.append(best_conf1)
                original_labels1.append(best_label1)

                # Fix label assignment for algo1
                reviewed_labels1.append("P" if best_iou1 > 0.2 and best_conf1 >= 0.7 else "N")
                if best_label1=="P" and best_iou1 > 0.2 and best_conf1 < 0.7: 
                    row["note"] = "Inconsistency: algo1 is 'P' but IoU > 0.2 and confidence < 0.7." 
                elif best_label1=="P" and best_iou1 <= 0.2:  
                    row["note"] = "Inconsistency: algo1 is 'P' but IoU < 0.2."

            # Process IoU and confidence for algo2
            for gt_box in gt_bboxes:
                best_iou2 = 0
                best_conf2 = 0
                best_label2= "N"
                for algo2_box in algo2_bboxes:
                    iou = calculate_iou(gt_box, algo2_box)
                    if iou > best_iou2:
                        best_iou2 = iou
                        best_conf2 = algo2_box[4]
                        # Original label assignment for algo2
                        best_label2 = algo2_label

                iou_values2.append(best_iou2)
                conf_values2.append(best_conf2)
                original_labels2.append(best_label2)

                # Fix label assignment for algo2
                reviewed_labels2.append("P" if best_iou2 > 0.2 and best_conf2 >= 0.5 else "N")
                if best_label2=="P" and best_iou2 > 0.2 and best_conf2 < 0.5: 
                    row["note"] = "Inconsistency: algo2 is 'P' but IoU > 0.2 and confidence < 0.5." 
                elif best_label2=="P" and best_iou2 <= 0.2:  
                    row["note"] = "Inconsistency: algo2 is 'P' but IoU < 0.2."

            # Add the metrics to the row
            row["iou_algo1"] = iou_values1
            row["conf_algo1"] = conf_values1
            row["iou_algo2"] = iou_values2
            row["conf_algo2"] = conf_values2
            row["original_label_algo1"] = original_labels1 
            row["reviewed_label_algo1"] = reviewed_labels1
            row["original_label_algo2"] = original_labels2
            row["reviewed_label_algo2"] = reviewed_labels2   

        data.append(row)

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save to Excel
    df.to_excel(output_excel_path, index=False)
    print(f"Excel file with metrics saved to {output_excel_path}")


def compute_metrics(X, y):
    """Compute ROC curve, AUC, confusion matrix, accuracy, precision, recall, and F1 score.
    Args:
        X: Predicted scores or labels
        y: True labels
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the ROC curve
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    cf_matrix = confusion_matrix(y, X)
    accuracy = accuracy_score(y, X)
    precision = precision_score(y, X)
    recall = recall_score(y, X)
    f1 = f1_score(y, X)

    return fpr, tpr, roc_auc, accuracy, precision, recall, f1, cf_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process predictions JSON and generate metrics.")
    parser.add_argument("--input_json_path", type=str, required=True, help="Path to the predictions JSON file.")
    parser.add_argument("--combined_csv_path", type=str, required=True, help="Path to the combined_predictions.csv file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the outputs file.")
    parser.add_argument("--output_excel_path", type=str, required=True, help="Path to save the output Excel file.")
    parser.add_argument("--update_csv", action="store_true", help="Update combined_predictions.csv with JSON data.")
    parser.add_argument("--generate_boxplot", action="store_true", help="Generate boxplot and perform statistical test from combined_predictions.csv.")
    parser.add_argument("--generate_scatter_plot", action="store_true", help="Generate scatter plot combining IoU and confidence with color-coded edges.")
    parser.add_argument("--generate_excel", action="store_true", help="Generate Excel file with all IoU and confidence values.")
    parser.add_argument("--generate_excel_with_metrics", action="store_true", help="Generate Excel file with additional metrics.")

    args = parser.parse_args()

    # Load and process the JSON data
    json_data = load_json(args.input_json_path)

    # Update combined_predictions.csv with JSON data if specified
    if args.update_csv:
        update_combined_predictions(json_data, args.combined_csv_path)

    # Generate boxplot and perform statistical test if specified
    if args.generate_boxplot:
        generate_boxplot_and_test_from_csv(args.combined_csv_path, args.output_dir)

    # Generate scatter plot if specified
    if args.generate_scatter_plot:
        generate_scatter_plot_iou_confidence_with_colored_edges(args.combined_csv_path, args.output_dir)

    # Generate Excel file with all IoU and confidence values if specified
    if args.generate_excel:
        generate_excel_from_csv(args.combined_csv_path, args.output_excel_path)

    # Generate Excel file with additional metrics if specified
    if args.generate_excel_with_metrics:
        generate_excel_with_metrics(args.combined_csv_path, args.output_excel_path)
