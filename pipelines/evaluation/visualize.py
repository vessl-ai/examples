import os
import json
import plotly.express as px
import pandas as pd

# Read eval output json file
def read_json_from_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        
        if not json_files:
            print("No Json file")
            return None
        
        json_file_path = os.path.join(directory_path, json_files[0])
        
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Read {json_files[0]}")
            return data
    
    except Exception as e:
        print(f"Error: {e}")
        return None

origin_result_data = read_json_from_directory("/root/eval_output/origin/__root__original_model")
finetuned_result_data = read_json_from_directory("/root/eval_output/finetuned/__root__finetuned_model")

origin_model_name = os.environ['ORIGINAL_MODEL_REGISTRY']
finetuned_model_name = os.environ['FINETUNED_MODEL_REGISTRY']

def mmlu_case(origin_v, finetuned_v):
    acc, stderr = {}, {}
    if 'acc,none' in origin_v:
        acc[origin_model_name] = origin_v['acc,none']
        acc[finetuned_model_name] = finetuned_v['acc,none']
    elif 'exact_match,none' in origin_v:
        acc[origin_model_name] = origin_v['exact_match,none']
        acc[finetuned_model_name] = finetuned_v['exact_match,none']
    else:
        acc[origin_model_name] = None
        acc[finetuned_model_name] = None

    if 'acc_stderr,none' in origin_v:
        stderr[origin_model_name] = origin_v['acc_stderr,none']
        stderr[finetuned_model_name] = finetuned_v['acc_stderr,none']
    elif 'exact_match_stderr,none' in origin_v:
        stderr[origin_model_name] = origin_v['exact_match_stderr,none']
        stderr[finetuned_model_name] = finetuned_v['exact_match_stderr,none']
    else:
        stderr[origin_model_name] = None
        stderr[finetuned_model_name] = None
    return acc, stderr

def ifeval_case(task, origin_v, finetuned_v):
    def append_metrics(target_dict, metrics):
        for metric in metrics:
            if metric in origin_v:
                origin_result = origin_v[metric]
                finetuned_result = finetuned_v[metric]
                target_dict[f"{task}_{metric.replace(',none', '').replace('_acc','').replace('_stderr','')}"] = {
                    origin_model_name: origin_result,
                    finetuned_model_name: finetuned_result
                }
        return target_dict
            
    acc, stderr = {}, {}
    metrics_acc = [
        'inst_level_loose_acc,none',
        'inst_level_strict_acc,none',
        'prompt_level_loose_acc,none',
        'prompt_level_strict_acc,none'
    ]
    
    metrics_stderr = [
        'inst_level_loose_acc_stderr,none',
        'inst_level_strict_acc_stderr,none',
        'prompt_level_loose_acc_stderr,none',
        'prompt_level_strict_acc_stderr,none'
    ]

    return append_metrics(acc, metrics_acc), append_metrics(stderr, metrics_stderr)

acc_dict = {}
stderr_dict = {}

task_list = list(origin_result_data['results'].keys())
for task in task_list:
    origin_v = origin_result_data['results'][task]
    finetuned_v = finetuned_result_data['results'][task]
    
    # mmlu case
    if task.startswith("mmlu"):
        acc_dict[f"{task}"], stderr_dict[f"{task}"] = mmlu_case(origin_v, finetuned_v)

    # ifeval case
    if task.startswith("ifeval"):
        ifeval_acc, ifeval_stderr = ifeval_case(task, origin_v, finetuned_v)
        acc_dict.update(ifeval_acc)
        stderr_dict.update(ifeval_stderr)

df_acc = pd.DataFrame.from_dict(acc_dict).T.reset_index(names=['task'])
df_stderr = pd.DataFrame.from_dict(stderr_dict).T.reset_index(names=['task'])
df_result = pd.merge(df_acc, df_stderr, on='task', suffixes=('_acc', '_stderr'))

df_acc.to_csv("/root/eval_output/accuracy_result.csv", index=False)
df_stderr.to_csv("/root/eval_output/stderr_result.csv", index=False)
df_result.to_csv("/root/eval_output/merged_result.csv", index=False)
