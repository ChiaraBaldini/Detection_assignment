# Detection Assignment

This project is designed to process and analyze predictions from different algorithms (algo1 and algo2), compare their performance, and generate various metrics, plots and reports.

## Project Structure

```
Detection_Assignment/
|
├── cpp_output.csv
├── combined_cpp_prediction.zip (csv file updated with both AI team and Software team predictions)
├── combined_predictions_with_metrics.xlsx 
├── data_distribution_analysis.py
├── process_cpp_outputs_and_compare.py
├── process_predictions_json.py
├── README.md
├── results.xlsx
└── Figs/
```

## Requirements

- Python 3.9 or higher
- Required Python libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy

## Scripts Overview

### 1. `process_cpp_outputs_and_compare.py`
This script processes the `cpp_output.csv` file, extracting predictions and metadata, calculates detection metrics for algo1 and algo2, and compares the performance of two algorithms across different subgroups (sex, brand etc.)

#### Usage:
```
python process_cpp_outputs_and_compare.py \
    --csv_file_path <path_to_cpp_output.csv> \
    --subgroup_column "sex"" \
    --gt_column "GT" \
    --algo_column "algo1" \
    --group1 "Male" \
    --group2 "Female" \
    --output_dir <output_directory> \
    --algo_column2 "algo2"
```              


### 2. `process_predictions_json.py`
This script processes a JSON file containing predictions from the AI team (bounding boxes), updates the `combined_predictions.csv` file, and generates Intersection over Union (IoU) metrics by comparing them with ground-truth boxes, BB distribution visualizations, and a single Excel file with all the extracted information.

#### Usage:
```
python process_predictions_json.py \
    --input_json_path <path_to_predictions.json> \
    --combined_csv_path <path_to_combined_predictions.csv> \
    --output_excel_path <path_to_output_excel.xlsx> \
    [--update_csv] [--generate_boxplot] [--generate_scatter_plot] [--generate_excel] [--generate_excel_with_metrics]
```

## How to Run
1. Ensure all required Python libraries are installed.
2. Prepare the input files (`cpp_output.csv`, `predictions.json`, etc.).
3. Run the desired script with the appropriate arguments.
4. Check the output directory for generated files and visualizations ("Figs" folder).

## Contact
For any questions or issues, please contact chiara.baldini16@gmail.com.