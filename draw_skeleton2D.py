import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import cv2
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoFileClip
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from PIL import Image, ImageSequence
import argparse
import configparser
import yaml
matplotlib.use('Agg')

skeleton_body_part =    [   [[2, 3], [2, 4], [1, 2]],          # skeleton_bone
                            [[0, 1], [1, 3], [0, 2], [2, 4]],  # skeleton_middle
                            [[5, 7], [7, 9]],                  # skeleton_left_hand
                            [[6, 8], [8, 10]],                 # skeleton_left_hand
                            [[5, 11], [11, 13], [13, 15]],
                            [[6, 12], [12, 14], [14, 16]]]     # skeleton_right_hand

combinations = [[i, j] for i in range(0, 22) for j in range(i+1, 22)]
# alpha_pose 1 will map to SMPL 12 
# alpha_pose 2 will map to SMPL 15
# alpha_pose 5 will map to SMPL 16
# alpha_pose 6 will map to SMPL 17
#                     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,11,12,13,14,15,16       # AlphaPose
AlphaPose_to_SMPL = [-1, 12, 15, -1, -1, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]      # SMPL

#                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21  # SMPL
SMPL_to_AlphaPose = [-1,11,12,-1,13,14,-1,15,16,-1,-1,-1, 1,-1,-1, 2, 5, 6, 7, 8, 9,10] # AlphaPose

# not include 0 3 6 9 10 11 13 14
not_include = []
for i in range(0, 22):
    if i not in AlphaPose_to_SMPL:
        not_include.append(i)

SMPL_joints_name = ["pelvis",       "left hip",     "right hip",        "spine 1",      "left knee", 
                    "right knee",   "spine 2",      "left ankle",       "right ankle",  "spine 3", 
                    "left foot",    "right foot",   "neck",             "left collar",  "right collar", 
                    "head",         "left shoulder","right shoulder",   "left elbow",   "right elbow", 
                    "left wrist",   "right wrist"]

def readattention(video_name, num_length, attention_id, attention_node_path, attention_matrix_path):
    with open(attention_node_path) as f:
        attention_node = json.load(f)
    with open(attention_matrix_path) as f:
        matrix_file = json.load(f)

    attention_node = np.array(attention_node[video_name])
    attention_matrix_new = np.array(matrix_file[video_name][attention_id])
    attention_node_len = len(attention_node[0])
    color_node = []
    for i in range(0, num_length, 1):
        index = int(i * (attention_node_len / num_length))
        color_node.append(attention_node[0, index])
    
    return color_node, attention_matrix_new

def Top3Node(video_name, attention_node_path, output_path):
    with open(attention_node_path) as f:
        attention_node = json.load(f)
        
    attention_node = np.array(attention_node[video_name])
    accumulate_node = np.zeros(22)

    for nodes in attention_node[0]:
        indexmax = np.argsort(-nodes)
        accumulate_node[indexmax[0]]+=3
        accumulate_node[indexmax[1]]+=2
        accumulate_node[indexmax[2]]+=1

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

def draw(joints, output_path, file_name, video_path_dir, attention_node_path, attention_matrix_path):

    video_path = video_path_dir +'/alpha_pose_' + file_name + "/" + file_name + ".mp4" 
    output_path_origin = output_path
    output_gif_path = os.path.join(output_path, file_name + '.gif')

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = 30
    frame_index = 0
    num_frame = len(joints)
    output, gif = [], []

    ''' Run all frame of a video '''
    while (frame_index != num_frame):
        _, frame_c = cap.read()
        if not _:
            print('Finish',file_name)
            break 

        ''' Four attention matrix '''
        concatenate_frame = []
        for attention_id in [0,1,2,3]:
            color_node, attention_matrix_new = readattention(file_name, num_frame, attention_id, attention_node_path, attention_matrix_path)
            frame = frame_c.copy()
            COLORMAP = cv2.COLORMAP_SPRING
                    
            ''' 
            Attention Node 
            '''
            # Attention node of according frame_index
            colornodes = np.array(color_node[frame_index])
    
            # Sort the SMPL node in descending order 
            indexmax = np.argsort(-colornodes)

            for i in range(0, len(indexmax)):
                if (indexmax[i] in not_include):
                    colornodes[indexmax[i]] = -1

            # Sort the SMPL node that can mapped to Alphapose in descending order 
            indexmax = np.argsort(-colornodes)

            x_coordinates, y_coordinates = joints[frame_index][:,0], joints[frame_index][:,1]

            Top3Node = []
            for i in range(0, 3):
                node = {}
                node['SMPL_index'] = indexmax[i]
                node['AlphaPose_index'] = SMPL_to_AlphaPose[indexmax[i]]  
                node['value'] = colornodes[indexmax[i]]
                node['rank'] = i
                node['x'], node['y'] = x_coordinates[node['AlphaPose_index']],  y_coordinates[node['AlphaPose_index']]
                Top3Node.append(node)
  
            for node in Top3Node:
                '''
                Setting color weight according by attention id 
                Color map will be reberse the color weight (ex COLORMAP_SPRING & COLORMAP_AUTUMN) 
                '''
                color_map = cv2.applyColorMap(np.array([[0 + 127 * node['rank']]]).astype(np.uint8), COLORMAP)
                rgb_value = (int(color_map[0, 0, 0]), int(color_map[0, 0, 1]), int(color_map[0, 0, 2]))
                cv2.circle(frame, (int(node['x']), int(node['y'])), int(14 - node['rank']*3), rgb_value, -1)
          
            ''' 
            Attention Link 
            '''
            all_link = []
            for SMPL_pair in combinations:
                SMPL_A,SMPL_B = SMPL_pair[0], SMPL_pair[1]
                if(SMPL_to_AlphaPose[SMPL_A] != -1 and SMPL_to_AlphaPose[SMPL_B] != -1):
                    link = {}
                    link['SMPL_pair'] = SMPL_pair
                    link['AlphaPose_pair'] = [SMPL_to_AlphaPose[SMPL_A], SMPL_to_AlphaPose[SMPL_B]]
                    link['value'] = attention_matrix_new[SMPL_A][SMPL_B] + attention_matrix_new[SMPL_B][SMPL_A]
                    link['A_x'], link['A_y'] = x_coordinates[SMPL_to_AlphaPose[SMPL_A]], y_coordinates[SMPL_to_AlphaPose[SMPL_A]]
                    link['B_x'], link['B_y'] = x_coordinates[SMPL_to_AlphaPose[SMPL_B]], y_coordinates[SMPL_to_AlphaPose[SMPL_B]]                                                                                    
                    all_link.append(link)
            
            all_link.sort(key=lambda x: x['value'], reverse=True)
            Top3Link = all_link[:3]
       
            for i in range(0, 3):
                Top3Link[i]['rank'] = i

            for link in Top3Link:
                pt1, pt2 = (int(link['A_x']), int(link['A_y'])) , (int(link['B_x']), int(link['B_y']))
                '''
                Setting color weight according by attention id 
                Color map will be reberse the color weight (ex COLORMAP_SPRING & COLORMAP_AUTUMN) 
                '''
                color_map = cv2.applyColorMap(np.array([[0 + 127 * link['rank']]]).astype(np.uint8), COLORMAP)
                rgb_value = (int(color_map[0, 0, 0]), int(color_map[0, 0, 1]), int(color_map[0, 0, 2]))
                cv2.line(frame, pt1, pt2, rgb_value, 10 - 3*link['rank'])
            
            concatenate_frame.append(frame)
   
        frame = np.concatenate(([i for i in concatenate_frame]), axis=1)
        gif = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        gif = Image.fromarray(gif)
        output.append(gif)

        frame_index += 1

    # Take msec as unit, fps is N. Duration means the time of each frame. Thus duration = 1000 / N.
    fps = 15
            
    if not os.path.exists(output_path_origin + '/' + file_name):
        os.makedirs(output_path_origin + '/' + file_name)

    # save all single frame to png the path folder
    for i in range(len(output)):
        output[i].save(output_path_origin + '/' + file_name + '/' + file_name + '_'+ str(i) + '.png')
    output[0].save(output_gif_path, save_all=True, append_images=output[1:], loop=0, disposal=2, duration=1000/fps)
    cap.release()
    cv2.destroyAllWindows()

def read_file(file_path):
    with open(file_path, "rb") as f:
        labeldata = json.load(f)
    joints = list()
    for body in labeldata:
        keypoints_matrix = np.array(body["keypoints"]).reshape((17, 3))[:, :2]
        joints.append(keypoints_matrix)
    return joints

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    attention_node = config['attention_node']
    attention_matrix = config['attention_matrix']
    epoch_num  = config['epoch_num']
    output_dir = config['output_dir']
    video_dir = config['video_dir']
    return attention_node, attention_matrix, epoch_num, output_dir, video_dir

def findfilename(attention_node_path):
    with open(attention_node_path) as f:
        attention_node = json.load(f)
    All_filenames = []
    for item in attention_node:
        All_filenames.append(item)
    return All_filenames

# Use command
# python draw_skeleton2D.py /home/weihsin/projects/Evaluation/config_file/finetuneAttention.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read configuration file and run setup.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    attention_node, attention_matrix, epoch_num, output_dir, video_dir = read_config(args.config_path)

    # When running the error segment setting, it is must to check the file name in the attention_node file
    All_filenames = findfilename(attention_node)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(os.path.dirname(file_path))
                file_name = file_name.split('e_')[1]
                if file_name in All_filenames :
                    print(file_name)
                    joints = read_file(file_path)
                    draw(joints, output_dir, file_name, video_dir,attention_node ,attention_matrix)
                    Top3Node(file_name, attention_node, output_dir)
                    Top3Link(file_name, attention_matrix, output_dir)