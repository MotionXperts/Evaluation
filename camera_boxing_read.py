import os, json, cv2, argparse, pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoFileClip
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from PIL import Image, ImageSequence
from drawutils.rank_color import skeleton_body_part, joint_graph_coordinate, AlphaPose_to_SMPL, SMPL_to_AlphaPose
from drawutils.rank_color import SMPL_joints_name, rank_color_list, backbone_color, combinations,AlphaPose_not_draw
from drawutils.read_file import read_file, read_config, findfilename
from drawutils.Top3 import readattention, Top3Node, Top3Link, readframe, readtext
offset = 0.3
matplotlib.use('Agg')

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def normal_draw(joints_cam, file_name):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    
    cam_list = ["1","2","3","4"]
    color = ['gold', 'cyan', 'red', 'lime']
    num_frame_min = 100000
    for idx, i in enumerate(cam_list):
        try:
            num_frame = len(joints_cam[i])
            num_frame_min = min(num_frame_min,num_frame)
            joints_cam[i] = joints_cam[i].reshape(num_frame,22,3)
            # y z 顛倒
            joints_cam[i][:, :, 1], joints_cam[i][:, :, 2] = -joints_cam[i][:, :, 1], -joints_cam[i][:, :, 2]
        except:
            continue

    def animate(frame):
        ax.clear()

        for idx, i in enumerate(cam_list):
            try:
                joints = joints_cam[i]
            except:
                continue
            skeleton = joints[frame]
            ax.set_xlim([joints[0][0][0]-offset, joints[0][0][0]+offset])  # 设置 x 轴范围
            ax.set_ylim([joints[0][0][1]-offset, joints[0][0][1]+offset])  # 设置 y 轴范围
            ax.set_zlim([joints[0][0][2]-offset, joints[0][0][2]+offset])   # 设置 z 轴范围
            ax.set_xlabel('x axis')
            ax.set_zlabel('y axis')
            ax.set_ylabel('z axis')
    
            skeleton_bone = [[15,12],[12,9],[9,6],[6,3],[3,0]]
            skeleton_left_leg = [[0,1],[1,4],[4,7],[7,10]]
            skeleton_right_leg = [[0,2],[2,5],[5,8],[8,11]]
            skeleton_left_hand = [[9,13],[13,16],[16,18],[18,20]]
            skeleton_right_hand = [[9,14],[14,17],[17,19],[19,21]]
            
            for index in skeleton_bone : 
                first = index[0]
                second = index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]] , 
                        [skeleton[first][1], skeleton[second][1]] ,
                        [skeleton[first][2], skeleton[second][2]] , color[idx], linewidth=5.0)   # gold
            
            for index in skeleton_left_leg : 
                first = index[0]
                second = index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]] , 
                        [skeleton[first][1], skeleton[second][1]] ,
                        [skeleton[first][2], skeleton[second][2]] , color[idx], linewidth=5.0)  # cyan
        
            for index in skeleton_right_leg : 
                first = index[0]
                second = index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]] , 
                        [skeleton[first][1], skeleton[second][1]] ,
                        [skeleton[first][2], skeleton[second][2]] , color[idx], linewidth=5.0) # fuchsia
            
            for index in skeleton_left_hand : 
                first = index[0]
                second = index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]] , 
                        [skeleton[first][1], skeleton[second][1]] ,
                        [skeleton[first][2], skeleton[second][2]] , color[idx], linewidth=5.0) # lime
    
            for index in skeleton_right_hand : 
                first = index[0]
                second = index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]] , 
                        [skeleton[first][1], skeleton[second][1]] ,
                        [skeleton[first][2], skeleton[second][2]] , color[idx],linewidth=5.0) # red
        
        ax.grid(True)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ani = animation.FuncAnimation(fig, animate, frames=num_frame_min, repeat=True)
    gif = file_name + '.gif' 
    gif_dir = "./Boxing_cam_skeleton"
    gif_path = os.path.join(gif_dir,gif)
    print("gif_path",gif_path)
    ani.save(gif_path, fps=100)

# Use command
# python camera_boxing_read.py
if __name__ == "__main__":
    # train boxing path 
    train_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_GT_train_aggregate.pkl"
    # test boxing path
    test_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_GT_test_aggregate.pkl"
    train_dataset = load_pkl(train_path)
    test_dataset = load_pkl(test_path)

    # When running the error segment setting, it is must to check the file name in the attention_node file
   
    data_dict = {}
    for data in train_dataset : 
        video_name = data['video_name']
        prefix = video_name.split('_cam')[0]
        postfix = video_name.split('_cam')[1]
        if prefix not in data_dict:
            data_dict[prefix] = {}
        data_dict[prefix][postfix] = data["features"]
        data_dict[prefix]["name"] = prefix

    for data in test_dataset : 
        video_name = data['video_name']
        prefix = video_name.split('_cam')[0]
        postfix = video_name.split('_cam')[1]
        if prefix not in data_dict:
            data_dict[prefix] = {}
        data_dict[prefix][postfix] = data["features"]
        data_dict[prefix]["name"] = prefix

    for data in data_dict :
        normal_draw(data_dict[data], data_dict[data]["name"])