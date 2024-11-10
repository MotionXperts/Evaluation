import shutil
import os
import json
import pickle
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

def read_ground_truth(pkl_file,input_file,ground_json_file):
    with open(pkl_file, 'rb') as file:
        datas = pickle.load(file)
    GT = {}
    for data in datas:
        print(data)
        GT[data['video_name']] = data['labels']
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    filenames = []
    for line in lines:
        filename = line.strip() 
        filenames.append(filename)
    jsonoutput = {}
    with  open(ground_json_file, 'w') as outfile:
        for filename in filenames:
            jsonoutput[filename] = GT[filename]
        json.dump(jsonoutput, outfile, indent=4)

input_file  = 'golden_filename_boxing'
pkl_file    = '/home/c1l1mo/datasets/boxing_safetrim/boxing_GT_test/aggregate.pkl'
input_dir   =           '/home/weihsin/projects/Evaluation/finetuneBoxing'
output_dir  =           '/home/weihsin/projects/Evaluation/package/finetuneBoxing'
input_json_path =       '/home/andrewchen/MotionExpert_v2/MotionExpert/results/finetune_skeleton_boxing/jsons/results_epoch20.json'
output_json_path =      '/home/weihsin/projects/Evaluation/package/finetuneBoxing/results.json'
ground_json_file =      '/home/weihsin/projects/Evaluation/package/finetuneBoxing/ground_truth.json'
remove_lines_with_gif(input_file, input_dir, output_dir)
move_results(input_file,input_json_path, output_json_path)
read_ground_truth(pkl_file,input_file,ground_json_file)