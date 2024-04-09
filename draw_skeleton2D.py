# draw skeleton to mp4 video
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random, os, json, matplotlib, cv2
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoFileClip
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from PIL import Image,ImageSequence
folder_path = '/home/weihsin/datasets/Skating_Alphapose'
vidoe_path = '/home/weihsin/datasets/Skating_Clip_Axel_All'
new_images_path = '/home/weihsin/datasets/Skating_Clip_Axel_All/{}.jpg'
matplotlib.use('Agg')
attention_node_path   = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune/att_node_results_epoch81.json'
attention_matrix_path = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune/att_A_results_epoch81.json'
skeleton_body_part = [[[2, 3], [2, 4], [1, 2]],          # skeleton_bone
                      [[0, 1], [1, 3], [0, 2], [2, 4]],  # skeleton_middle
                      [[5, 7], [7, 9]],                  # skeleton_left_hand
                      [[6, 8], [8, 10]],                 # skeleton_left_hand
                      [[5,11], [11, 13], [13, 15]],
                      [[6, 12], [12, 14], [14,16]]]      # skeleton_right_hand

attention_id = 3
filenames = [   "467205307287470390_0", 
                "467205310373953653_0", 
                "467205326496858477_0", 
                "467205329533534401_0", 
                "467205339415314713_0", 
                "471706283780080147_2",
                "471706363236974645_0"]
          # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16                    
mapping = [ -1, 12,15,-1,-1,16,17,18,19,20, 21,  1,  2,  4,  5,  7,  8]
not_include = []
for i in range(0,22) :
    if i not in mapping : 
        not_include.append(i)

combinations = [[i,j] for i in range(0,17) for j in range(0,17) ]
def readattention(video_name,num_length):
    with open(attention_node_path) as f:         attention_node   = json.load(f)
    with open(attention_matrix_path) as f:       attention_matrix = json.load(f)
    attention_node = np.array(attention_node[video_name])
    attention_matrix_new = np.array(attention_matrix[video_name][attention_id])
    
    color_node = []

    for i in range(0,num_length,1 ):
        if num_length <= 131 :
            # 取(i/num_length)*160的 floor值
            index = int(i*(num_length/131))
            color_node.append(attention_node[0,index])
        if num_length > 131 :
            index = int(i*(131/num_length))
            color_node.append(attention_node[0,index])

    
    
    return color_node,attention_matrix_new

def draw(joints,video_path,output_path,file_name):    
    video_path = os.path.join(vidoe_path, file_name + '.mp4')

    cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = os.path.join(output_path, file_name + '.gif')
 
    frame_index = 0
    num_frame = len(joints)
    color_node,attention_matrix_new = readattention(file_name,num_frame) 

    threshold = 0.95
    max = np.max(attention_matrix_new)
   
    output = []  
    gif = []
    if attention_id == 0 :
        COLORMAP = cv2.COLORMAP_SPRING
    if attention_id == 1 :
        COLORMAP = cv2.COLORMAP_SUMMER
    if attention_id == 2 :
        COLORMAP = cv2.COLORMAP_COOL
    if attention_id == 3 : 
        COLORMAP = cv2.COLORMAP_AUTUMN
    while True:
        _, frame = cap.read()
        if not _: break
        colornodes = color_node[frame_index]
        
        min_score = colornodes.min() 
        max_score = colornodes.max() 
        
        colornodes = np.array(colornodes)
        indexmax = colornodes.argsort()
        weight = 0
        for id in indexmax:
            if(id in not_include) : continue
            colornodes[id] = (12 - weight) / 12
            weight += 1
        #########################################################################
        x_coordinates, y_coordinates = joints[frame_index][:, 0], joints[frame_index][:, 1]
        points_list = list(zip(x_coordinates, y_coordinates))
        #print(len(points_list))
        point_id = 0
        for point in points_list:
            if(mapping[point_id] == -1) : 
                point_id += 1
                continue

            scaled_value_uint8 = int((1-colornodes[mapping[point_id]]) * 255)

            color_map = cv2.applyColorMap(np.array([[scaled_value_uint8]]).astype(np.uint8),COLORMAP)
            rgb_value = (int(color_map[0, 0, 0]), int(color_map[0, 0, 1]), int(color_map[0, 0, 2]))
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, rgb_value, -1)
            point_id += 1
  
        top3 = []
        for index in combinations:
            first, second = index[0], index[1]
            if mapping[first] != -1 and mapping[second] != -1 and attention_matrix_new[mapping[first]][mapping[second]] > threshold :
                top3.append([attention_matrix_new[mapping[first]][mapping[second]],first,second])

        top3.sort(reverse=True)
        top3 = top3[:3]
    
        for index in top3:
            index[0] *= 100000

        rangeimportance = top3[0][0] - top3[2][0]
        smallest = top3[2][0]
        count = [1,0.5,0]

        count_weight = 0
        for index in top3:
            index[0] = count[count_weight]#(index[0] - smallest / rangeimportance)
            count_weight += 1

        for index in top3:
            first ,second = index[1] , index[2]   
            pt1 = (int(joints[frame_index][first][0]), int(joints[frame_index][first][1]))
            pt2 = (int(joints[frame_index][second][0]), int(joints[frame_index][second][1]))   
            
            scaled_value_uint8 = int((1-index[0]) * 255)
            color_map = cv2.applyColorMap(np.array([[scaled_value_uint8]]).astype(np.uint8), COLORMAP)
            rgb_value = (int(color_map[0, 0, 0]), int(color_map[0, 0, 1]), int(color_map[0, 0, 2]))
                
            cv2.line(frame,pt1,pt2,rgb_value,1)   
          
        frame_index += 1
        gif = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)  
        gif = Image.fromarray(gif)                   
    
        output.append(gif)                           

        if(frame_index == num_frame) :
            break

    output[0].save(output_path, save_all=True, append_images=output[1:], loop=0, disposal=2)
    cap.release()
    cv2.destroyAllWindows()

def extract_x_y(bodys) :
    joint = list()
    for body in bodys :
        keypoints_matrix = np.array(body["keypoints"]).reshape((17, 3))[:, :2]
        joint.append(keypoints_matrix)
    return joint

def read_file(file_path,file_name,output_folder) :
    with open(file_path,"rb") as f:
        labeldata = json.load(f)
    joints = extract_x_y(labeldata)
    
    draw(joints,file_path,output_folder,file_name)

def read_folder(folder_path,output_folder) :
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(os.path.dirname(file_path))
                file_name = file_name.split('e_')[1]
                if (file_name in filenames) :
                   read_file(file_path,file_name,output_folder)
                #read_file(file_path,file_name,output_folder)
 
output_folder = '/home/weihsin/projects/Evaluation/Video/attention'+str(attention_id)
read_folder(folder_path,output_folder)                
        