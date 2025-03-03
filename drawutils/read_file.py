import yaml, json
import numpy as np
def read_file(file_path):
    with open(file_path, "rb") as f:
        labeldata = json.load(f)
    joints = list()
    exists = {}
    for body in labeldata:
        if body["image_id"] not in exists: 
            exists[body["image_id"]] = True
            keypoints_matrix = np.array(body["keypoints"]).reshape((17, 3))[:, :2]
            joints.append(keypoints_matrix)
    return joints

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    attention_node      = config['attention_node']
    attention_matrix    = config['attention_matrix']
    index_dict          = config['index_dict']
    max_index           = config['max_index']
    epoch_num           = config['epoch_num']
    output_dir          = config['output_dir']
    video_dir           = config['video_dir']
    result_pth          = config['results_epoch']
    if 'golden_file' in config:
        golden_file     = config['golden_file']
    else:
        golden_file     = None
    return attention_node, attention_matrix, epoch_num, output_dir, video_dir, golden_file, index_dict, max_index, result_pth

def findfilename(attention_node_path):
    with open(attention_node_path) as f:
        attention_node = json.load(f)
    All_filenames = []
    for item in attention_node:
        All_filenames.append(item)
    return All_filenames