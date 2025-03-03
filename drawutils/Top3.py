import json, os
import numpy as np
from drawutils.rank_color import SMPL_joints_name, combinations
def readtext(file_name, results_pth):
    with open(results_pth) as f:
        texts  = json.load(f)
    return texts[file_name]
def readframe(file_name, index_dict_path, max_index_path):
    with open(index_dict_path) as f:
        index_dict  = json.load(f)
    with open(max_index_path) as f:
        max_index     = json.load(f)

    key_frame =  np.array(max_index[file_name])
    start_end_frame = index_dict[file_name]
    return key_frame, start_end_frame
def readattention(video_name, num_length, attention_id, attention_node_path, attention_matrix_path):
    with open(attention_node_path) as f:
        attention_node  = json.load(f)
    with open(attention_matrix_path) as f:
        matrix_file     = json.load(f)

    attention_node = np.array(attention_node[video_name])
    attention_matrix_new = np.array(matrix_file[video_name][attention_id])

    attention_node_len = len(attention_node[0])
    color_node = []
    for i in range(0, len(attention_node[0]), 1):
        color_node.append(attention_node[0, i])

    return color_node, attention_matrix_new

def Top3Node(video_name, attention_node_path, output_path):
    with open(attention_node_path) as f:
        attention_node = json.load(f)

    attention_node = np.array(attention_node[video_name])
    accumulate_node = np.zeros(22)

    for nodes in attention_node[0]:
        indexmax = np.argsort(-nodes)
        accumulate_node[indexmax[0]] += 3
        accumulate_node[indexmax[1]] += 2
        accumulate_node[indexmax[2]] += 1

    Top3joint = np.argsort(-accumulate_node)[:3]

    attention_node_save = {}
    rank = 1
    for joint in Top3joint:
        attention_node_save['rank_'+str(rank)] = SMPL_joints_name[joint]
        rank += 1

    if not os.path.exists(output_path + '/attention_node'):
        os.makedirs(output_path + '/attention_node')
    path = output_path + '/attention_node/' + video_name + '_node.json'
    with open(path, 'w') as f:
        json.dump(attention_node_save, f, indent = 1)

def Top3Link(video_name, attention_matrix_path, output_path):
    with open(attention_matrix_path) as f:
        matrix_file = json.load(f)

    attention = {}
    for attention_id in range(0, 4):

        attention_matrix = np.array(matrix_file[video_name][attention_id])
        all_link = []
        for SMPL_pair in combinations: 
            link = {}                                                                             
            SMPL_A,SMPL_B = SMPL_pair[0], SMPL_pair[1]
            link['SMPL_pair'] = SMPL_pair
            link['value'] = attention_matrix[SMPL_A][SMPL_B] + attention_matrix[SMPL_B][SMPL_A]
            all_link.append(link)

        all_link.sort(key=lambda x: x['value'], reverse=True)
        Top3Link = all_link[:3]

        rank = 1
        attention_matrix_save = {}
        for link in Top3Link:
            attention_matrix_save['rank_'+str(rank)] = [SMPL_joints_name[link['SMPL_pair'][0]], SMPL_joints_name[link['SMPL_pair'][1]]]
            rank += 1

        attention[str(attention_id)] = attention_matrix_save
    if not os.path.exists(output_path + '/attention_matrix'):
        os.makedirs(output_path + '/attention_matrix')
    path = output_path + '/attention_matrix/' + video_name + '_matrix.json'
    with open(path, 'w') as f:
        json.dump(attention, f, indent = 1)
