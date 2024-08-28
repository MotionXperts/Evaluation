import shutil
import os
import json
def remove_lines_with_gif(input_file, input_dir, output_dir):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        
    for line in lines:
        filename = line.strip() +'.gif'
        print(filename)
        input_filepath = os.path.join(input_dir, filename)
        target_filepath = os.path.join(output_dir, filename)
        print(input_filepath)
        print(target_filepath)
        shutil.copy(input_filepath, target_filepath)

def move_results(input_file, input_json_path, output_json_path):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    filenames = []
    for line in lines:
        filename = line.strip() 
        filenames.append(filename)

    jsonoutput = {}

    with open(input_json_path, 'r') as file:
        json_datas = json.load(file)

    for data in json_datas:
        if(data in filenames):
            jsonoutput[data] = json_datas[data]

    with open(output_json_path, 'w') as outfile:
        json.dump(jsonoutput, outfile, indent=4)

input_file = 'golden_filename'
input_dir =                     '/home/weihsin/projects/Evaluation/finetuneAttention_RGBdifference'
output_dir =            '/home/weihsin/projects/Evaluation/package/finetuneAttention_RGBdifference'
input_json_path =               '/home/weihsin/projects/Evaluation/finetuneAttention_RGBdifference/results_epoch70.json'
output_json_path =      '/home/weihsin/projects/Evaluation/package/finetuneAttention_RGBdifference/results.json'
remove_lines_with_gif(input_file, input_dir, output_dir)
move_results(input_file,input_json_path, output_json_path)