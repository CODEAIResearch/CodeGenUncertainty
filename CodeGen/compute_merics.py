
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
from scipy.integrate import trapezoid

# Load the JSON files   deepseek_coder_67b_instruct_hf  CodeLlama_7b_Instruct_hf_hf
metrics_file = "./evalplus_results/humaneval/home--rsr200002--LLMs--CodeLlama_7b_Instruct_hf_hf_temp_0.0.metrics.jsonl"
eval_results_file = "./evalplus_results/humaneval/home--rsr200002--LLMs--CodeLlama_7b_Instruct_hf_hf_temp_0.0.eval_results.json"

with open(metrics_file, "r") as f:
    metrics_data = [json.loads(line) for line in f.readlines()]

with open(eval_results_file, "r") as f:
    eval_results_data = json.load(f)

# Prepare data for AUC, PR-AUC, and Brier score calculations
results = []

# Create a dictionary for easy access to metrics by task_id
metrics_by_task_id = {entry["task_id"]: entry["uncertainty"] for entry in metrics_data}
#metrics_by_task_id = {
#    entry["task_id"]: {k: v for k, v in entry["uncertainty"].items() if k != "UCPMI"}
#    for entry in metrics_data
#}

# Sigmoid function to convert raw scores to probabilities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize a dictionary to accumulate true labels and predicted scores for each metric
accumulated_data = {metric: {"y_true_base": [], "y_true_plus": [], "y_score_base": [], "y_score_plus": []} for metric in metrics_data[0]["uncertainty"].keys()}

# Function to compute AUARC
def compute_auarc(y_true, y_score, thresholds=np.linspace(0, 1, 100)):
    accuracy = []
    for threshold in thresholds:
        # Apply threshold to reject predictions with low confidence
        accepted = y_score >= threshold
        correct = np.sum(np.array(y_true)[accepted] == 1)  # Correct predictions for the accepted samples
        total = np.sum(accepted)  # Total accepted predictions
        
        if total > 0:
            accuracy.append(correct / total)
        else:
            accuracy.append(0)
    
    # Compute area under the accuracy-rejection curve (AUARC) using the trapezoidal rule
    auarc = trapezoid(accuracy, thresholds)
    return auarc

# Iterate through each task in the eval results
for task_info in eval_results_data["eval"].values():
    if len(task_info) == 0:
        continue  # Skip if no results exist for this task
    task_results = task_info[0]  # Each task is a list, get the first entry

    task_id = task_results["task_id"]

    # Find the corresponding task's metrics using the task_id
    task_metrics = metrics_by_task_id.get(task_id)
    if not task_metrics:
        print(f"Task ID {task_id} not found in metrics data")
        continue

    # Extract results for both plus and base status
    base_status = task_results.get("base_status")
    plus_status = task_results.get("plus_status")

    # Convert statuses to 0 or 1 (0 = pass, 1 = fail)
    base_status_score = 0 if base_status == "pass" else 1
    plus_status_score = 0 if plus_status == "pass" else 1
    
    # For each metric, accumulate the true labels and predicted scores
    for metric_name, metric_value in task_metrics.items():
        # Apply sigmoid to convert raw scores to probabilities
        predicted_probability = sigmoid(metric_value)
        
        # Accumulate the true labels and predicted probabilities for base_status and plus_status
        accumulated_data[metric_name]["y_true_base"].append(base_status_score)
        accumulated_data[metric_name]["y_true_plus"].append(plus_status_score)
        accumulated_data[metric_name]["y_score_base"].append(predicted_probability)
        accumulated_data[metric_name]["y_score_plus"].append(predicted_probability)

# Now calculate AUC, PR-AUC, and Brier score for each metric
for metric_name, data in accumulated_data.items():
    # Compute ROC-AUC for base_status
    try:
        if len(set(data["y_true_base"])) > 1:  # Ensure both classes are present
            auc_base = roc_auc_score(data["y_true_base"], data["y_score_base"])
        else:
            auc_base = None
    except UndefinedMetricWarning:
        auc_base = None

    # Compute ROC-AUC for plus_status
    try:
        if len(set(data["y_true_plus"])) > 1:  # Ensure both classes are present
            auc_plus = roc_auc_score(data["y_true_plus"], data["y_score_plus"])
        else:
            auc_plus = None
    except UndefinedMetricWarning:
        auc_plus = None

    # Compute PR-AUC for base_status
    try:
        if len(set(data["y_true_base"])) > 1:
            pr_auc_base = average_precision_score(data["y_true_base"], data["y_score_base"])
        else:
            pr_auc_base = None
    except UndefinedMetricWarning:
        pr_auc_base = None

    # Compute PR-AUC for plus_status
    try:
        if len(set(data["y_true_plus"])) > 1:
            pr_auc_plus = average_precision_score(data["y_true_plus"], data["y_score_plus"])
        else:
            pr_auc_plus = None
    except UndefinedMetricWarning:
        pr_auc_plus = None

    #added here
    if len(data["y_true_base"]) > 0 and len(data["y_score_base"]) > 0:
        brier_base = brier_score_loss(data["y_true_base"], data["y_score_base"])
    else:
        brier_base = 0.0 

    # Compute Brier Score for base_status
    #brier_base = brier_score_loss(data["y_true_base"], data["y_score_base"])

    # Compute Brier Score for plus_status
    if len(data["y_true_plus"]) > 0 and len(data["y_score_plus"]) > 0:
        brier_plus = brier_score_loss(data["y_true_plus"], data["y_score_plus"])
    else:
        brier_plus = 0.0 
    #brier_plus = brier_score_loss(data["y_true_plus"], data["y_score_plus"])

    # Compute AUARC for base_status
    auarc_base = compute_auarc(data["y_true_base"], data["y_score_base"])

    # Compute AUARC for plus_status
    auarc_plus = compute_auarc(data["y_true_plus"], data["y_score_plus"])

    # Round results to 3 decimal places
    auc_base = round(auc_base, 3) if auc_base is not None else None
    auc_plus = round(auc_plus, 3) if auc_plus is not None else None
    pr_auc_base = round(pr_auc_base, 3) if pr_auc_base is not None else None
    pr_auc_plus = round(pr_auc_plus, 3) if pr_auc_plus is not None else None
    brier_base = round(brier_base, 3)
    brier_plus = round(brier_plus, 3)
    auarc_base = round(auarc_base, 3)
    auarc_plus = round(auarc_plus, 3)

    # Store results in the results list
    results.append({
        "metric_name": metric_name,
        "base_status_auc": auc_base,
        "plus_status_auc": auc_plus,
        "base_status_pr_auc": pr_auc_base,
        "plus_status_pr_auc": pr_auc_plus,
        "base_status_brier_score": brier_base,
        "plus_status_brier_score": brier_plus,
        "base_status_auarc": auarc_base,
        "plus_status_auarc": auarc_plus,
    })

# Create a DataFrame to store and visualize the results
results_df = pd.DataFrame(results)
# Save the results to a CSV file
results_df.to_csv("evaluation_metrics_results.csv", index=False)

print(results_df)

















"""import json
import pandas as pd
import os

import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

def resolve_paths(samples_path):
    if not samples_path.endswith(".jsonl"):
        raise ValueError("Expected a .jsonl file")

    base = samples_path[:-6]  # remove ".jsonl"
    eval_path = (
        base + "_eval_results.json"
        if os.path.exists(base + "_eval_results.json")
        else base + ".eval_results.json"
    )
    metrics_path = base + ".metrics.jsonl"

    return eval_path, metrics_path

def load_eval(eval_path):
    with open(eval_path, "r") as f:
        return json.load(f)

def load_metrics(metrics_path):
    return pd.read_json(metrics_path, lines=True)



def compute_ece(confidences, labels, n_bins=10):
    
    Compute Expected Calibration Error (ECE).
    - confidences: array of model confidence scores (0 to 1)
    - labels: array of binary correctness labels (0 or 1)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_indices = np.digitize(confidences, bins, right=True) - 1

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_size = bin_mask.sum()
        if bin_size == 0:
            continue
        bin_conf = confidences[bin_mask].mean()
        bin_acc = labels[bin_mask].mean()
        ece += np.abs(bin_acc - bin_conf) * (bin_size / len(confidences))
    return ece

if __name__ == "__main__":
    samples_path = "./evalplus_results/humaneval/home--rsr200002--LLMs--CodeLlama_7b_Instruct_hf_hf_temp_0.0.jsonl"

    eval_path, metrics_path = resolve_paths(samples_path)

    print(f"[+] Eval Results: {eval_path}")
    print(f"[+] Metrics File: {metrics_path}")

    eval_data = load_eval(eval_path)
    metrics_df = load_metrics(metrics_path)

    print(f"[✓] Loaded {len(eval_data['eval'])} tasks from eval results")
    print(f"[✓] Loaded {len(metrics_df)} uncertainty entries")


    # 1. Flatten eval_data into DataFrame
    eval_rows = []
    for task_id, completions in eval_data["eval"].items():
        assert len(completions) == 1, f"Multiple samples for task {task_id}; expected one"
        result = completions[0]
        eval_rows.append({
        "task_id": task_id,
        "base_correct": int(result["base_status"] == "pass"),
        "plus_correct": int(result["plus_status"] == "pass") if "plus_status" in result else None
    })

    df_eval = pd.DataFrame(eval_rows)

# 2. Flatten metrics
    uncertainty_flat = pd.json_normalize(metrics_df["uncertainty"])
    df_metrics = pd.concat([metrics_df[["task_id"]], uncertainty_flat], axis=1)

# 3. Merge on task_id
    df_all = pd.merge(df_eval, df_metrics, on="task_id")    

    # 4. Compute AUCs
    from sklearn.metrics import roc_auc_score

    metrics = ["MSP", "UPerp", "P_bar", "UHT", "TokenSAR", "PMI", "CPMI", "URD", "UFR"]

    print("🔍 AUC (base_correct as label):")
    for m in metrics:
        try:
            scores = -df_all[m] if m == "MSP" else df_all[m]  # MSP is confidence
            auc = roc_auc_score(df_all["base_correct"], scores)
            print(f"{m:10s}: AUC = {auc:.4f}")
        except Exception as e:
            print(f"{m:10s}: ERROR - {e}")


    print("\n📏 ECE (Expected Calibration Error):")
    for m in metrics:
        try:
        # MSP is a confidence score, others are uncertainty → invert
            confidence = df_all[m] if m == "MSP" else 1.0 - df_all[m]
            ece = compute_ece(confidence.values, df_all["base_correct"].values, n_bins=10)
            print(f"{m:10s}: ECE = {ece:.4f}")
        except Exception as e:
            print(f"{m:10s}: ERROR - {e}")

    print("\n🔍 AUC (plus_correct as label):")
    for m in metrics:
        try:
            scores = -df_all[m] if m == "MSP" else df_all[m]  # MSP is confidence
            auc = roc_auc_score(df_all["plus_correct"].dropna(), scores[df_all["plus_correct"].notna()])
            print(f"{m:10s}: AUC = {auc:.4f}")
        except Exception as e:
            print(f"{m:10s}: ERROR - {e}")

    print("\n📏 ECE (plus_correct):")
    for m in metrics:
        try:
            confidence = df_all[m] if m == "MSP" else 1.0 - df_all[m]
            mask = df_all["plus_correct"].notna()
            ece = compute_ece(confidence[mask].values, df_all["plus_correct"][mask].values, n_bins=10)
            print(f"{m:10s}: ECE = {ece:.4f}")
        except Exception as e:
            print(f"{m:10s}: ERROR - {e}")"""


