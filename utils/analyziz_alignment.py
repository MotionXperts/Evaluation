import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
model_colors = {"No_diff": "blue", "RGB": "green", "Skeleton": "red"}

# Load data
with open("/home/weihsin/projects/Evaluation/utils/few_shot_organized.json", "r") as file:
    data = json.load(file)

kendalls_tau_best_model = {"No_diff": [], "RGB": [], "Skeleton": []}
kendalls_tau_worst_model = {"No_diff": [], "RGB": [], "Skeleton": []}

# Define metrics in lowercase for consistency
metrics = ["bleu_1", "bleu_4", "rouge", "cider", "bertscore"]
kendalls_tau_best = {metric: {"No_diff": [], "RGB": [], "Skeleton": []} for metric in metrics}
kendalls_tau_worst = {metric: {"No_diff": [], "RGB": [], "Skeleton": []} for metric in metrics}

# Parse JSON data
for entry in data:
    kendall_value = data[entry]["KendallsTau"]
    best_model = data[entry]["CoachMe"]["best_model"]
    worst_model = data[entry]["CoachMe"]["worst_model"]
    
    if kendall_value is not None:
        # Append Kendall's Tau values to best and worst models
        if best_model in kendalls_tau_best_model:
            kendalls_tau_best_model[best_model].append(kendall_value)

        if worst_model in kendalls_tau_worst_model:
            kendalls_tau_worst_model[worst_model].append(kendall_value)
            
        # Append metric scores to best and worst models for each metric
        for metric in metrics:
            if best_model in kendalls_tau_best[metric]:
                kendalls_tau_best[metric][best_model].append((kendall_value, data[entry]["CoachMe"]["scores_best"][metric]))
            if worst_model in kendalls_tau_worst[metric]:
                kendalls_tau_worst[metric][worst_model].append((kendall_value, data[entry]["CoachMe"]["scores_worst"][metric]))

# Plot Kendall's Tau vs each metric for each model
def plot_kendalls_tau_vs_metric_for_model(metric, model):
    plt.figure(figsize=(10, 6))

    # Best model points
    tau_best, score_best = zip(*kendalls_tau_best[metric][model]) if kendalls_tau_best[metric][model] else ([], [])
    plt.scatter(tau_best, score_best, color=model_colors[model], label=f"Best - {model}", marker='o', alpha=0.6)

    # Worst model points
    tau_worst, score_worst = zip(*kendalls_tau_worst[metric][model]) if kendalls_tau_worst[metric][model] else ([], [])
    plt.scatter(tau_worst, score_worst, color=model_colors[model], label=f"Worst - {model}", marker='x', alpha=0.6)

    # Labels and title
    plt.xlabel("Kendall's Tau")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f"{metric.capitalize()} vs Kendall's Tau for {model}")

    # Save plot
    plt.savefig(f"{metric}_kendalls_tau_analysis_{model}.png")
    plt.close()

# Generate plots for each metric and each model
for metric in metrics:
    for model in model_colors:
        plot_kendalls_tau_vs_metric_for_model(metric, model)

def linear_regression_analysis(metric):
    # Prepare data for regression: combine best and worst model data points for all models
    tau_best = []
    score_best = []
    tau_worst = []
    score_worst = []
    
    for model in kendalls_tau_best[metric]:
        tau_best.extend([x[0] for x in kendalls_tau_best[metric][model]])
        score_best.extend([x[1] for x in kendalls_tau_best[metric][model]])
        
    for model in kendalls_tau_worst[metric]:
        tau_worst.extend([x[0] for x in kendalls_tau_worst[metric][model]])
        score_worst.extend([x[1] for x in kendalls_tau_worst[metric][model]])
    
    # Combine all data points
    tau = np.array(tau_best + tau_worst).reshape(-1, 1)
    score = np.array(score_best + score_worst).reshape(-1, 1)
    
    if len(tau) > 1 and len(score) > 1:
        # Perform linear regression
        model_lr = LinearRegression().fit(tau, score)

        # Get the slope (coefficient) and intercept
        slope = model_lr.coef_[0][0]
        intercept = model_lr.intercept_[0]

        # Return the slope value for analysis
        return slope
    else:
        return None
    
# Perform linear regression for each metric and print the slope values
slope_results = {}
for metric in metrics:
    slope_results[metric] = linear_regression_analysis(metric)

# Output the results for analysis
print("Results (the linear regression slope β between Kendall's Tau and metrics for all models):")
for metric, slope in slope_results.items():
    if slope is not None:
        if slope == 0:
            print(f"{metric.capitalize()}: No correlation between Kendall's Tau and {metric}.")
        else:
            print(f"{metric.capitalize()}: Slope = {slope:.2f}")
            print(f"{metric.capitalize()}: Kendall's Tau and {metric} have a {'positive' if slope > 0 else 'negative'} correlation.")
    else:
        print(f"{metric.capitalize()}: Not enough data for linear regression.")

# Correlation analysis for Kendall's Tau vs each metric (all models combined)
def correlation_analysis(metric):
    # Prepare data for correlation: combine best and worst model data points for all models
    tau_best = []
    score_best = []
    tau_worst = []
    score_worst = []
    
    for model in kendalls_tau_best[metric]:
        if model == 'Skeleton' :
            tau_best.extend([x[0] for x in kendalls_tau_best[metric][model]])
            score_best.extend([x[1] for x in kendalls_tau_best[metric][model]])
        
    for model in kendalls_tau_worst[metric]:
        if model == 'Skeleton' :
            tau_worst.extend([x[0] for x in kendalls_tau_worst[metric][model]])
            score_worst.extend([x[1] for x in kendalls_tau_worst[metric][model]])
    
    # Combine all data points
    tau = np.array(tau_best + tau_worst)
    score = np.array(score_best + score_worst)
    
    if len(tau) > 1 and len(score) > 1:
        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(tau, score)  # Pearson's correlation coefficient
        return correlation
    else:
        return None

# Perform correlation analysis for each metric and print the correlation coefficient
correlation_results = {}
for metric in metrics:
    print(metric)
    correlation_results[metric] = correlation_analysis(metric)

# Output the results for analysis
print("Results (the Pearson correlation coefficient between Kendall's Tau and metrics for all models):")
for metric, correlation in correlation_results.items():
    if correlation is not None:
        if abs(correlation) < 0.1:
            print(f"{metric.capitalize()}: Very weak or no correlation between Kendall's Tau and {metric}.")
        elif correlation > 0:
            print(f"{metric.capitalize()}: Positive correlation = {correlation:.2f}")
        elif correlation < 0:
            print(f"{metric.capitalize()}: Negative correlation = {correlation:.2f}")
        else:
            print(f"{metric.capitalize()}: No correlation between Kendall's Tau and {metric}.")
    else:
        print(f"{metric.capitalize()}: Not enough data for correlation analysis.")