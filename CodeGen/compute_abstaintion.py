import json
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import pearsonr
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import matplotlib.pyplot as plt
import os

# Load the JSON files   deepseek_coder_67b_instruct_hf  CodeLlama_7b_Instruct_hf_hf
metrics_file = "./evalplus_results/humaneval/home--rsr200002--LLMs--CodeLlama_7b_Instruct_hf_hf_temp_0.0.metrics.jsonl"
eval_results_file = "./evalplus_results/humaneval/home--rsr200002--LLMs--CodeLlama_7b_Instruct_hf_hf_temp_0.0.eval_results.json"


with open(metrics_file, "r") as f:
    metrics_data = [json.loads(line) for line in f.readlines()]

with open(eval_results_file, "r") as f:
    eval_results_data = json.load(f)

# Create a dictionary for easy access to metrics by task_id
metrics_by_task_id = {entry["task_id"]: entry["uncertainty"] for entry in metrics_data}
#metrics_by_task_id = {
#    entry["task_id"]: {k: v for k, v in entry["uncertainty"].items() if k != "UCPMI"}
#    for entry in metrics_data
#}

# Prepare data for each metric
accumulated_data = {metric: {"y_true_base": [], "y_true_plus": [], "y_score": []} 
                    for metric in metrics_data[0]["uncertainty"].keys()}

# Collecting data
for task_info in eval_results_data["eval"].values():
    if len(task_info) == 0:
        continue  # Skip if no results exist for this task
    task_results = task_info[0]  # Each task is a list, get the first entry

    task_id = task_results["task_id"]
    task_metrics = metrics_by_task_id.get(task_id)

    # Extract results for both plus and base status
    base_status = task_results.get("base_status")
    plus_status = task_results.get("plus_status")

    # Convert statuses to 0 or 1 (0 = pass, 1 = fail)
    base_status_score = 0 if base_status == "pass" else 1
    plus_status_score = 0 if plus_status == "pass" else 1

    for metric_name, metric_value in task_metrics.items():
        # Directly use raw scores (no sigmoid)
        predicted_score = metric_value  # This is a raw score, not a probability

        # Accumulate the true labels and predicted scores
        accumulated_data[metric_name]["y_true_base"].append(base_status_score)
        accumulated_data[metric_name]["y_true_plus"].append(plus_status_score)
        accumulated_data[metric_name]["y_score"].append(predicted_score)

# Store results for best threshold, mispredictions, and total samples
best_results = []

# Define a minimum sample threshold (to avoid trivial cases)
min_samples_for_consideration = 0.20  # Minimum 20% of total samples in the identified group

# Sweep over a range of thresholds (dynamically between min and max scores)
for metric_name, data in accumulated_data.items():
    if len(data["y_score"]) > 0:
        thresholds = np.linspace(np.min(data["y_score"]), np.max(data["y_score"]), 100)
    else:
        thresholds = np.array([0.0])
    #thresholds = np.linspace(np.min(data["y_score"]), np.max(data["y_score"]), 100)  # Dynamic threshold range

    best_threshold = None
    best_misprediction_percentage_base = 0
    best_misprediction_percentage_plus = 0
    best_mispredictions_count_base = 0
    best_mispredictions_count_plus = 0
    best_total_samples_count_base = 0
    best_total_samples_count_plus = 0

    y_true_base = np.array(data["y_true_base"])
    y_true_plus = np.array(data["y_true_plus"])
    y_score = np.array(data["y_score"])

    total_samples = len(y_true_base)  # Total number of samples

    for theta in thresholds:
        # Identify samples above and below the threshold
        above_threshold = y_score >= theta
        below_threshold = ~above_threshold

        # Count mispredictions (fail status) above and below the threshold for base_status
        mispredictions_above_base = np.sum((y_true_base[above_threshold] == 1))  # Mispredictions for base_status (fail)
        total_samples_above = np.sum(above_threshold)

        mispredictions_below_base = np.sum((y_true_base[below_threshold] == 1))  # Mispredictions for base_status (fail)
        total_samples_below = np.sum(below_threshold)

        # Count mispredictions (fail status) above and below the threshold for plus_status
        mispredictions_above_plus = np.sum((y_true_plus[above_threshold] == 1))  # Mispredictions for plus_status (fail)
        total_samples_above_plus = np.sum(above_threshold)

        mispredictions_below_plus = np.sum((y_true_plus[below_threshold] == 1))  # Mispredictions for plus_status (fail)
        total_samples_below_plus = np.sum(below_threshold)

        # Calculate misprediction percentages for base_status above and below the threshold
        if total_samples_above > 0:
            misprediction_percentage_above_base = (mispredictions_above_base / total_samples_above) * 100
        else:
            misprediction_percentage_above_base = 0

        if total_samples_below > 0:
            misprediction_percentage_below_base = (mispredictions_below_base / total_samples_below) * 100
        else:
            misprediction_percentage_below_base = 0

        # Calculate misprediction percentages for plus_status above and below the threshold
        if total_samples_above_plus > 0:
            misprediction_percentage_above_plus = (mispredictions_above_plus / total_samples_above_plus) * 100
        else:
            misprediction_percentage_above_plus = 0

        if total_samples_below_plus > 0:
            misprediction_percentage_below_plus = (mispredictions_below_plus / total_samples_below_plus) * 100
        else:
            misprediction_percentage_below_plus = 0

        # Apply the minimum sample threshold (percentage of total samples)
        min_samples = min_samples_for_consideration * total_samples
        if total_samples_above >= min_samples and misprediction_percentage_above_base > best_misprediction_percentage_base:
            best_threshold = theta
            best_misprediction_percentage_base = misprediction_percentage_above_base
            best_mispredictions_count_base = mispredictions_above_base
            best_total_samples_count_base = total_samples_above

        if total_samples_below >= min_samples and misprediction_percentage_below_base > best_misprediction_percentage_base:
            best_threshold = theta
            best_misprediction_percentage_base = misprediction_percentage_below_base
            best_mispredictions_count_base = mispredictions_below_base
            best_total_samples_count_base = total_samples_below

        if total_samples_above_plus >= min_samples and misprediction_percentage_above_plus > best_misprediction_percentage_plus:
            best_threshold = theta
            best_misprediction_percentage_plus = misprediction_percentage_above_plus
            best_mispredictions_count_plus = mispredictions_above_plus
            best_total_samples_count_plus = total_samples_above_plus

        if total_samples_below_plus >= min_samples and misprediction_percentage_below_plus > best_misprediction_percentage_plus:
            best_threshold = theta
            best_misprediction_percentage_plus = misprediction_percentage_below_plus
            best_mispredictions_count_plus = mispredictions_below_plus
            best_total_samples_count_plus = total_samples_below_plus

    # Save results for best threshold, mispredictions, and total samples for the current metric
    #if best_threshold is not None:
    #    best_threshold = round(best_threshold, 3)
    #else:
    #    best_threshold = 0.0
    best_results.append({
        "metric_name": metric_name,
        "best_threshold": best_threshold,
        "best_misprediction_percentage_base": round(best_misprediction_percentage_base, 3),
        "best_mispredictions_count_base": best_mispredictions_count_base,
        "best_total_samples_count_base": best_total_samples_count_base,
        "best_misprediction_percentage_plus": round(best_misprediction_percentage_plus, 3),
        "best_mispredictions_count_plus": best_mispredictions_count_plus,
        "best_total_samples_count_plus": best_total_samples_count_plus
    })

    # Print the best results for the current metric
    """print(f"Metric: {metric_name}")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Base Status Misprediction Percentage: {best_misprediction_percentage_base}%")
    print(f"Best Base Status Mispredictions Count: {best_mispredictions_count_base}")
    print(f"Best Base Status Total Samples Count: {best_total_samples_count_base}")
    print(f"Best Plus Status Misprediction Percentage: {best_misprediction_percentage_plus}%")
    print(f"Best Plus Status Mispredictions Count: {best_mispredictions_count_plus}")
    print(f"Best Plus Status Total Samples Count: {best_total_samples_count_plus}")
    print("\n")"""

# Optionally, create a DataFrame to display the best threshold results
best_results_df = pd.DataFrame(best_results)
best_results_df.to_csv("abstaintion_results.csv", index=False)
print(best_results_df)


"""
Maximum Sequence Probability 
Perplexity 
Mean Token Entropy 
Pointwise Mutual Information 
Conditional Pointwise Mutual Information 
Rényi Divergence 
Fisher-Rao Distance 
TokenSAR 
Claim-Conditioned Probability
Monte Carlo Sequence Entropy
Semantic Entropy
SentenceSAR
Number of Semantic Sets
Sum of Eigenvalues of the Graph Laplacian
Lexical Similarity
BB Semantic Entropy
LabelProb
"""