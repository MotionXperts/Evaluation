import argparse
import yaml
import pickle
import json
SMPL_joints_name = ["pelvis",       "left hip",     "right hip",        "spine 1",      "left knee", 
                    "right knee",   "spine 2",      "left ankle",       "right ankle",  "spine 3", 
                    "left foot",    "right foot",   "neck",             "left collar",  "right collar", 
                    "head",         "left shoulder","right shoulder",   "left elbow",   "right elbow", 
                    "left wrist",   "right wrist"]

def read_matrix(matrix_path):
    with open(matrix_path) as f:
        matrix_file = json.load(f)

    attention_id = ['0','1','2','3']
    rank = ['rank_1', 'rank_2', 'rank_3']
    for id in attention_id:
        for r in rank:
            if ( ' 1' in matrix_file[id][r][0] or ' 2' in matrix_file[id][r][0] or ' 3' in matrix_file[id][r][0]):
                matrix_file[id][r][0]  = matrix_file[id][r][0][:-2]
            if ( ' 1' in matrix_file[id][r][1] or ' 2' in matrix_file[id][r][1] or ' 3' in matrix_file[id][r][1]):
                matrix_file[id][r][1]  = matrix_file[id][r][1][:-2]

    return matrix_file    

def read_node(node_path):
    with open(node_path) as f:
        node_file = json.load(f)

    rank = ['rank_1', 'rank_2', 'rank_3']
    for r in rank:
        if ( ' 1' in node_file[r] or ' 2' in node_file[r] or ' 3' in node_file[r]):
            node_file[r]  = node_file[r][:-2]
        if ( ' 1' in node_file[r] or ' 2' in node_file[r] or ' 3' in node_file[r]):
            node_file[r]  = node_file[r][:-2]
    return node_file

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = config['output_dir']
    label_json = config['results_epoch']
    return output_dir, label_json
    
# Use command
# python utils/GPTprepare.py /home/weihsin/projects/Evaluation/config_file/finetuneAttention.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read configuration file and run setup.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    output_dir, label_json = read_config(args.config_path)
    attention_matrix_dir = output_dir + '/attention_matrix'
    attention_node_dir = output_dir + '/attention_node'
    output_path = output_dir + '/gpt_input.json'

    with open(label_json) as f:
        data_entries = json.load(f)

    error_segment_not_use = [   '471706304466387043_2',
                                '471703098860503282_1',
                                '495776032394576594_1',
                                '471703104162365684_1',
                                '495776032394576594_0',
                                '471706290155159700_1',
                                '485958841751044210_0',
                                '471706304466387043_1',
                                '471706263479386249_1' ]

    datas = []
    for entry in data_entries:
        data = {}
        print(entry)
        data['video_name'] = entry
        if data['video_name'] in error_segment_not_use:
            continue
        attention_matrix_path = attention_matrix_dir + '/' + data['video_name'] + '_matrix.json'
        matrix = read_matrix(attention_matrix_path)

        attention_node_path = attention_node_dir + '/' + data['video_name'] + '_node.json'
        node = read_node(attention_node_path)

        data['matrix'] = matrix
        data['node'] = node
        data['instruction'] = data_entries[entry]
        datas.append(data)

    with open(output_path, 'w') as f:
        json.dump(datas, f, indent=4)
