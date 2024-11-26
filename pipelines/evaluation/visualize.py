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

output_directory = "/root/eval_output/__root__model" 
json_data = read_json_from_directory(output_directory)

# Generate acc,stderr result dataframe
tasks = []
acc = []
stderr = []

for i, v in json_data['results'].items():
    tasks.append(i)
    if 'acc,none' in v:
        acc.append(v['acc,none'])
    elif 'exact_match,none' in v:
        acc.append(v['exact_match,none'])
    else:
        acc.append(0)

    if 'acc_stderr,none' in v:
        stderr.append(v['acc_stderr,none'])
    elif 'exact_match_stderr,none' in v:
        stderr.append(v['exact_match_stderr,none'])
    else:
        stderr.append(0)
df = pd.DataFrame({"tasks": tasks, "acc": acc, "stderr": stderr})

# Visualize acc,stderr result and save
fig = px.line_polar(df[['tasks', 'acc']], r='acc', theta='tasks', line_close=True, color_discrete_sequence=['green'])
fig.update_traces(fill='toself')
fig.write_image(f"{output_directory}/acc_result.png")
fig.show()

fig = px.line_polar(df[['tasks', 'stderr']], r='stderr', theta='tasks', line_close=True, color_discrete_sequence=['blue'])
fig.update_traces(fill='toself')
fig.write_image(f"{output_directory}/stderr_result.png")
fig.show()