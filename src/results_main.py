import json
import pandas as pd
import ast
import os
import math
import re
from datetime import datetime
from utils import hr_at_k, dcg_at_k, ndcg_at_k, PROJECT_PATH, model_name_url_list


def evaluate_topk(result_path, llmdd_path):
    """Evaluate localization results"""
    defect_results_file_path = os.path.join(result_path, "agent3_file_importance_scores.json")
    suspicious_paths = []
    with open(defect_results_file_path) as f:
        data = json.load(f)
        for item in data:
            file_path = item.get("file_path")
            if file_path and file_path not in suspicious_paths:
                suspicious_paths.append(file_path)

    # 2. Extract ID from path name
    folder_name = os.path.basename(result_path)
    first_id = int(folder_name.split("__")[0])
    # 3. Read llmdd_defect.csv to get correct paths
    df = pd.read_csv(llmdd_path)
    row = df[df["id"] == first_id].iloc[0]
    correct_paths = ast.literal_eval(row["correct source-code paths"])
    base_k = len(correct_paths)
    defect_type = row["types"]  # Get defect type

    evaluate_topk_results = {
        "project_id": first_id,
        "defect_type": defect_type,  # Add defect type
        "base_k": base_k,
        "base_k_hr": hr_at_k(suspicious_paths, correct_paths, base_k),
        "base_k_ndcg": ndcg_at_k(suspicious_paths, correct_paths, base_k),
        "base_k+2_hr": hr_at_k(suspicious_paths, correct_paths, base_k + 2),
        "base_k+2_ndcg": ndcg_at_k(suspicious_paths, correct_paths, base_k + 2),
        "base_k+4_hr": hr_at_k(suspicious_paths, correct_paths, base_k + 4),
        "base_k+4_ndcg": ndcg_at_k(suspicious_paths, correct_paths, base_k + 4),
        "suspicious_paths": suspicious_paths,
        "correct_paths": correct_paths,
    }
    return evaluate_topk_results


def summarize_llm_token_usage(llm_calls_path):
    """Summarize total token usage (prompt, completion, total) from calls_llm.jsonl"""
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    api_call_count = 0  # Add API call count

    with open(llm_calls_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            api_call_count += 1  # Each line represents one API call

            usage = None
            if "response" in data and isinstance(data["response"], dict) and "usage" in data["response"]:
                usage = data["response"]["usage"]
            elif "usage" in data:  # Fallback for other possible structures
                usage = data["usage"]

            if usage:
                total_tokens += usage.get("total_tokens", 0)
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)

    token_usage = {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "api_call_count": api_call_count,  # Add API call count to return results
    }
    return token_usage


def get_log_time_range(log_path):
    """Get start and end timestamps from running.log and compute duration in seconds"""
    time_format = "%Y-%m-%d %H:%M:%S,%f"
    start_time = None
    end_time = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
            if match:
                ts = datetime.strptime(match.group(1), time_format)
                if start_time is None:
                    start_time = ts
                end_time = ts
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()
    else:
        duration = None

    duration = {"duration": duration}
    return duration


if __name__ == "__main__":
    model_name_url_list = model_name_url_list
    llmdd_path = f"{PROJECT_PATH}/datasets/llmdd_defect.csv"
    results_root_path = f"{PROJECT_PATH}/results"
    result_dir_name = "main_output"
    # result_dir_name = "main_output_batch_20_250521_best"
    summary_results_path = os.path.join(results_root_path, result_dir_name, "summary_results.csv")

    # Add summary results path for defect subtypes
    defect_type_summary_path = os.path.join(results_root_path, result_dir_name, "defect_type_summary_results.csv")

    # Add summary results path for defect categories
    defect_category_summary_path = os.path.join(results_root_path, result_dir_name, "defect_category_summary_results.csv")

    # Import mapping from defect types to categories from motivation_base.py
    from motivation_base import llm_defect_categories

    # Create a mapping dictionary from defect types to categories
    type_to_category = {}
    for category, types_list in llm_defect_categories.items():
        for type_name in types_list:
            type_to_category[type_name] = category

    summary_metrics = []
    # For storing evaluation results by defect category
    all_category_metrics = []
    # For storing evaluation results by defect subtype
    all_type_metrics = []

    for model_name, _ in model_name_url_list:
        base_dir = f"{PROJECT_PATH}/results/{result_dir_name}/{model_name}"
        all_results_path = f"{PROJECT_PATH}/results/{result_dir_name}/all_results_{model_name}.csv"
        all_results_code_kgs_path = f"{PROJECT_PATH}/results/all_results_main_code_kgs_{model_name}.csv"
        all_results = []

        # Read all defect types
        df_llmdd = pd.read_csv(llmdd_path)

        # Iterate through each project directory
        for project_dir in os.listdir(base_dir):
            if project_dir == "_code_kgs":
                continue

            project_full_path = os.path.join(base_dir, project_dir)
            defect_results_json_path = os.path.join(project_full_path, "defect_results.json")
            calls_llm_jsonl_path = os.path.join(project_full_path, "calls_llm.jsonl")
            running_log_path = os.path.join(project_full_path, "running.log")

            if not os.path.exists(defect_results_json_path) or not os.path.exists(calls_llm_jsonl_path) or not os.path.exists(running_log_path):
                print(f"Skipping {project_full_path} because no results found")
                continue

            current_project_metrics = {}

            # Get evaluation metrics (now includes defect type)
            current_project_metrics = evaluate_topk(project_full_path, llmdd_path)

            # Get token usage and API call count
            token_usage_stats = summarize_llm_token_usage(calls_llm_jsonl_path)
            current_project_metrics.update(token_usage_stats)

            # Get runtime
            duration_stats = get_log_time_range(running_log_path)
            current_project_metrics.update(duration_stats)

            current_project_metrics["project_dir"] = project_dir
            all_results.append(current_project_metrics)

        df_all_projects = pd.DataFrame(all_results)
        # Sort and reset index
        df_all_projects = df_all_projects.sort_values(by="project_id")
        df_all_projects.to_csv(all_results_path, index=False)
        print(f"Saved detailed results for model {model_name} to {all_results_path}")

        # Add defect category column to df_all_projects
        if "defect_type" in df_all_projects.columns:
            df_all_projects["defect_category"] = df_all_projects["defect_type"].map(lambda x: type_to_category.get(x, "未分类"))

            # Calculate evaluation metrics grouped by defect subtype
            for defect_type in df_all_projects["defect_type"].unique():
                df_type = df_all_projects[df_all_projects["defect_type"] == defect_type]

                if len(df_type) == 0:
                    continue

                type_metric = {
                    "model_name": model_name,
                    "defect_type": defect_type,
                    "Total Projects": len(df_type),
                    # API call statistics
                    "Total API Calls": df_type["api_call_count"].sum() if "api_call_count" in df_type.columns else 0,
                    "Avg API Calls": df_type["api_call_count"].mean() if "api_call_count" in df_type.columns and df_type["api_call_count"].notna().any() else 0,
                    # Token statistics
                    "Total Tokens": df_type["total_tokens"].sum() if "total_tokens" in df_type.columns else 0,
                    "Prompt Tokens": df_type["prompt_tokens"].sum() if "prompt_tokens" in df_type.columns else 0,
                    "Completion Tokens": df_type["completion_tokens"].sum() if "completion_tokens" in df_type.columns else 0,
                    # Average tokens
                    "Avg Total Tokens": df_type["total_tokens"].mean() if "total_tokens" in df_type.columns and df_type["total_tokens"].notna().any() else 0,
                    "Avg Prompt Tokens": df_type["prompt_tokens"].mean() if "prompt_tokens" in df_type.columns and df_type["prompt_tokens"].notna().any() else 0,
                    "Avg Completion Tokens": df_type["completion_tokens"].mean() if "completion_tokens" in df_type.columns and df_type["completion_tokens"].notna().any() else 0,
                    # Runtime statistics
                    "Total Duration (seconds)": df_type["duration"].sum() if "duration" in df_type.columns and df_type["duration"].notna().any() else 0,
                    "Average Duration (seconds)": df_type["duration"].mean() if "duration" in df_type.columns and df_type["duration"].notna().any() else 0,
                    # Evaluation metrics
                    "Avg base_k HR": df_type["base_k_hr"].mean() if "base_k_hr" in df_type.columns and df_type["base_k_hr"].notna().any() else 0,
                    "Avg base_k NDCG": df_type["base_k_ndcg"].mean() if "base_k_ndcg" in df_type.columns and df_type["base_k_ndcg"].notna().any() else 0,
                    "Avg base_k+2 HR": df_type["base_k+2_hr"].mean() if "base_k+2_hr" in df_type.columns and df_type["base_k+2_hr"].notna().any() else 0,
                    "Avg base_k+2 NDCG": df_type["base_k+2_ndcg"].mean() if "base_k+2_ndcg" in df_type.columns and df_type["base_k+2_ndcg"].notna().any() else 0,
                    "Avg base_k+4 HR": df_type["base_k+4_hr"].mean() if "base_k+4_hr" in df_type.columns and df_type["base_k+4_hr"].notna().any() else 0,
                    "Avg base_k+4 NDCG": df_type["base_k+4_ndcg"].mean() if "base_k+4_ndcg" in df_type.columns and df_type["base_k+4_ndcg"].notna().any() else 0,
                }
                all_type_metrics.append(type_metric)

            # Calculate evaluation metrics grouped by defect category
            for category in llm_defect_categories.keys():
                df_category = df_all_projects[df_all_projects["defect_category"] == category]

                if len(df_category) == 0:
                    continue

                category_metric = {
                    "model_name": model_name,
                    "defect_category": category,
                    "Total Projects": len(df_category),
                    # API call statistics
                    "Total API Calls": df_category["api_call_count"].sum() if "api_call_count" in df_category.columns else 0,
                    "Avg API Calls": df_category["api_call_count"].mean() if "api_call_count" in df_category.columns and df_category["api_call_count"].notna().any() else 0,
                    # Token statistics
                    "Total Tokens": df_category["total_tokens"].sum() if "total_tokens" in df_category.columns else 0,
                    "Prompt Tokens": df_category["prompt_tokens"].sum() if "prompt_tokens" in df_category.columns else 0,
                    "Completion Tokens": df_category["completion_tokens"].sum() if "completion_tokens" in df_category.columns else 0,
                    # Average tokens
                    "Avg Total Tokens": df_category["total_tokens"].mean() if "total_tokens" in df_category.columns and df_category["total_tokens"].notna().any() else 0,
                    "Avg Prompt Tokens": df_category["prompt_tokens"].mean() if "prompt_tokens" in df_category.columns and df_category["prompt_tokens"].notna().any() else 0,
                    "Avg Completion Tokens": df_category["completion_tokens"].mean() if "completion_tokens" in df_category.columns and df_category["completion_tokens"].notna().any() else 0,
                    # Runtime statistics
                    "Total Duration (seconds)": df_category["duration"].sum() if "duration" in df_category.columns and df_category["duration"].notna().any() else 0,
                    "Average Duration (seconds)": df_category["duration"].mean() if "duration" in df_category.columns and df_category["duration"].notna().any() else 0,
                    # Evaluation metrics
                    "Avg base_k HR": df_category["base_k_hr"].mean() if "base_k_hr" in df_category.columns and df_category["base_k_hr"].notna().any() else 0,
                    "Avg base_k NDCG": df_category["base_k_ndcg"].mean() if "base_k_ndcg" in df_category.columns and df_category["base_k_ndcg"].notna().any() else 0,
                    "Avg base_k+2 HR": df_category["base_k+2_hr"].mean() if "base_k+2_hr" in df_category.columns and df_category["base_k+2_hr"].notna().any() else 0,
                    "Avg base_k+2 NDCG": df_category["base_k+2_ndcg"].mean() if "base_k+2_ndcg" in df_category.columns and df_category["base_k+2_ndcg"].notna().any() else 0,
                    "Avg base_k+4 HR": df_category["base_k+4_hr"].mean() if "base_k+4_hr" in df_category.columns and df_category["base_k+4_hr"].notna().any() else 0,
                    "Avg base_k+4 NDCG": df_category["base_k+4_ndcg"].mean() if "base_k+4_ndcg" in df_category.columns and df_category["base_k+4_ndcg"].notna().any() else 0,
                }
                all_category_metrics.append(category_metric)

        # Calculate overall metrics for the model (consistent with original code)
        summary_metrics_sub = {
            "model_name": model_name,
            "Total Projects": len(df_all_projects),
            # API call statistics
            "Total API Calls": df_all_projects["api_call_count"].sum() if "api_call_count" in df_all_projects.columns else 0,
            "Avg API Calls": df_all_projects["api_call_count"].mean() if "api_call_count" in df_all_projects.columns and df_all_projects["api_call_count"].notna().any() else 0,
            # Token statistics
            "Total Tokens": df_all_projects["total_tokens"].sum() if "total_tokens" in df_all_projects.columns else 0,
            "Prompt Tokens": df_all_projects["prompt_tokens"].sum() if "prompt_tokens" in df_all_projects.columns else 0,
            "Completion Tokens": df_all_projects["completion_tokens"].sum() if "completion_tokens" in df_all_projects.columns else 0,
            # Average tokens
            "Avg Total Tokens": df_all_projects["total_tokens"].mean() if "total_tokens" in df_all_projects.columns and df_all_projects["total_tokens"].notna().any() else 0,
            "Avg Prompt Tokens": df_all_projects["prompt_tokens"].mean() if "prompt_tokens" in df_all_projects.columns and df_all_projects["prompt_tokens"].notna().any() else 0,
            "Avg Completion Tokens": df_all_projects["completion_tokens"].mean() if "completion_tokens" in df_all_projects.columns and df_all_projects["completion_tokens"].notna().any() else 0,
            # Runtime statistics
            "Total Duration (seconds)": df_all_projects["duration"].sum() if "duration" in df_all_projects.columns and df_all_projects["duration"].notna().any() else 0,
            "Average Duration (seconds)": df_all_projects["duration"].mean() if "duration" in df_all_projects.columns and df_all_projects["duration"].notna().any() else 0,
            # Evaluation metrics
            "Avg base_k HR": df_all_projects["base_k_hr"].mean() if "base_k_hr" in df_all_projects.columns and df_all_projects["base_k_hr"].notna().any() else 0,
            "Avg base_k NDCG": df_all_projects["base_k_ndcg"].mean() if "base_k_ndcg" in df_all_projects.columns and df_all_projects["base_k_ndcg"].notna().any() else 0,
            "Avg base_k+2 HR": df_all_projects["base_k+2_hr"].mean() if "base_k+2_hr" in df_all_projects.columns and df_all_projects["base_k+2_hr"].notna().any() else 0,
            "Avg base_k+2 NDCG": df_all_projects["base_k+2_ndcg"].mean() if "base_k+2_ndcg" in df_all_projects.columns and df_all_projects["base_k+2_ndcg"].notna().any() else 0,
            "Avg base_k+4 HR": df_all_projects["base_k+4_hr"].mean() if "base_k+4_hr" in df_all_projects.columns and df_all_projects["base_k+4_hr"].notna().any() else 0,
            "Avg base_k+4 NDCG": df_all_projects["base_k+4_ndcg"].mean() if "base_k+4_ndcg" in df_all_projects.columns and df_all_projects["base_k+4_ndcg"].notna().any() else 0,
        }
        summary_metrics.append(summary_metrics_sub)

    # Save evaluation results by defect subtype
    df_type_metrics = pd.DataFrame(all_type_metrics)
    df_type_metrics = df_type_metrics.sort_values(by=["model_name", "defect_type"])
    df_type_metrics.to_csv(defect_type_summary_path, index=False)
    print(f"Saved defect type summary metrics to {defect_type_summary_path}")

    # Save evaluation results by defect category
    df_category_metrics = pd.DataFrame(all_category_metrics)
    df_category_metrics = df_category_metrics.sort_values(by=["model_name", "defect_category"])
    df_category_metrics.to_csv(defect_category_summary_path, index=False)
    print(f"Saved defect category summary metrics to {defect_category_summary_path}")

    # Save original summary metrics
    summary_metrics_df = pd.DataFrame(summary_metrics)
    summary_metrics_df = summary_metrics_df.sort_values(by="model_name")
    summary_metrics_df.to_csv(summary_results_path, index=False)
    print(f"Saved summary metrics to {summary_results_path}")
