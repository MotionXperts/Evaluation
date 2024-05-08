# draw skeleton to mp4 video
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import json
import cv2
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoFileClip
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from PIL import Image, ImageSequence
'''
Loop : 
dir_path = '/home/weihsin/datasets/Loop/clip_Loop_alphapose'
video_path_dir = '/home/weihsin/datasets/Loop/clip_Loop'
attention_node_path = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune_loop/att_node_results_epoch'
attention_matrix_path = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune_loop/att_A_results_epoch'
filenames = [   "467205977016893491_0",
                "467205981765107731_1",
                "467206017835860564_0",
                "467205981765107731_2",
                "467206001762763156_0",
                "467205970188828915_1",
                "467205973392753029_0",
                "467205970188828915_0",
                "467205981765107731_0",
                "467205977016893491_2"]

Axel :
dir_path = '/home/weihsin/datasets/Skating_Alphapose'
video_path_dir = '/home/weihsin/datasets/Skating_Clip_Axel_All'
attention_node_path = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune_new/att_node_results_epoch'
attention_matrix_path = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune_new/att_A_results_epoch'
output_dir = '/home/weihsin/projects/Evaluation/Video_Axel_new'
filenames = [   "467205307287470390_0",
                "467205310373953653_0",
                "467205326496858477_0",
                "467205329533534401_0",
                "467205339415314713_0",
                "471706283780080147_2",
                "471706363236974645_0"]
'''
# Path Setting 
dir_path = '/home/weihsin/datasets/Skating_Alphapose'
video_path_dir = '/home/weihsin/datasets/Skating_Clip_Axel_All'
attention_node = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune_new/att_node_results_epoch'
attention_matrix = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune_new/att_A_results_epoch'
output = '/home/weihsin/projects/Evaluation/Video_Axel_new'
filenames = [   "467205307287470390_0",
                "467205310373953653_0",
                "467205326496858477_0",
                "467205329533534401_0",
                "467205339415314713_0",
                "471706283780080147_2",
                "471706363236974645_0"]

matplotlib.use('Agg')
# Epoch start Setting
epoch = 80 

if not os.path.exists(output):
    os.makedirs(output)

for epoch in range(80, 85):
    output_dir = output +'/epoch' + str(epoch)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    attention_node_path = attention_node + str(epoch) + '.json'
    attention_matrix_path = attention_matrix + str(epoch) + '.json'

    skeleton_body_part = [[[2, 3], [2, 4], [1, 2]],          # skeleton_bone
                        [[0, 1], [1, 3], [0, 2], [2, 4]],  # skeleton_middle
                        [[5, 7], [7, 9]],                  # skeleton_left_hand
                        [[6, 8], [8, 10]],                 # skeleton_left_hand
                        [[5, 11], [11, 13], [13, 15]],
                        [[6, 12], [12, 14], [14, 16]]]      # skeleton_right_hand

    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    mapping = [-1, 12, 15, -1, -1, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
    not_include = []
    for i in range(0, 22):
        if i not in mapping:
            not_include.append(i)

    combinations = [[i, j] for i in range(0, 17) for j in range(0, 17)]


    def readattention(video_name, num_length, attention_id):
        with open(attention_node_path) as f:
            attention_node = json.load(f)
        with open(attention_matrix_path) as f:
            attention_matrix = json.load(f)
        attention_node = np.array(attention_node[video_name])
        attention_matrix_new = np.array(attention_matrix[video_name][attention_id])

        color_node = []
        for i in range(0, num_length, 1):
            if num_length <= 131:
                index = int(i * (num_length / 131))
                color_node.append(attention_node[0, index])
            if num_length > 131:
                index = int(i * (131 / num_length))
                color_node.append(attention_node[0, index])
        return color_node, attention_matrix_new


    def draw(joints, output_path, file_name):
        video_path_tmp = os.path.join(video_path_dir,file_name)
        video_path = video_path_dir +'/' + file_name + "/" + file_name + ".mp4" 
        output_path = os.path.join(output_path, file_name + '.gif')
        print(video_path)
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        fps = 30
        width, height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT))

        frame_index, threshold = 0, 0.95
        num_frame = len(joints)

        output, gif = [], []

        while True:
            _, frame_c = cap.read()
            if not _:
                print('Finish')
                break 
            concatenate_frame = []
            for attention_id in [0,1,2,3]:
                color_node, attention_matrix_new = readattention(
                    file_name, num_frame, attention_id)
                frame = frame_c.copy()
                if attention_id == 0:
                    COLORMAP = cv2.COLORMAP_SPRING
                if attention_id == 1:
                    COLORMAP = cv2.COLORMAP_SUMMER
                if attention_id == 2:
                    COLORMAP = cv2.COLORMAP_COOL
                if attention_id == 3:
                    COLORMAP = cv2.COLORMAP_AUTUMN

                colornodes = np.array(color_node[frame_index])
                indexmax = colornodes.argsort()
                weight = 0
                for id in indexmax:
                    if (id in not_include):
                        continue
                    colornodes[id] = (12 - weight) / 12
                    weight += 1
                x_coordinates, y_coordinates = joints[frame_index][:,0], joints[frame_index][:, 1]
                points_list = list(zip(x_coordinates, y_coordinates))
                point_id = 0
                for point in points_list:
                    if (mapping[point_id] == -1):
                        point_id += 1
                        continue
                    scaled_value_uint8 = int(
                        (1 - colornodes[mapping[point_id]]) * 255)
                    size = (1 - colornodes[mapping[point_id]]) * 12
                    # color_map = cv2.applyColorMap(np.array([[scaled_value_uint8]]).astype(np.uint8),COLORMAP)
                    if attention_id == 0 or attention_id == 3:
                        color_map = cv2.applyColorMap(
                            np.array([[0 + 127 * size]]).astype(np.uint8), COLORMAP)
                    else:
                        color_map = cv2.applyColorMap(
                            np.array([[255 - 127 * size]]).astype(np.uint8), COLORMAP)
                    rgb_value = (int(color_map[0, 0, 0]), int(
                        color_map[0, 0, 1]), int(color_map[0, 0, 2]))
                    if (size < 3):
                        cv2.circle(frame, (int(point[0]), int(point[1])), int(
                            4 - round(size, 1)), rgb_value, -1)
                    point_id += 1

                top3 = []
                for index in combinations:
                    first, second = index[0], index[1]
                    if mapping[first] != -1 and mapping[second] != - \
                            1 and attention_matrix_new[mapping[first]][mapping[second]] > threshold:
                        top3.append([attention_matrix_new[mapping[first]]
                                    [mapping[second]], first, second])

                top3.sort(reverse=True)
                top3 = top3[:3]
                count = [1, 0.5, 0]

                count_weight = 0
                for index in top3:
                    index[0] = count[count_weight]
                    count_weight += 1

                No1 = True
                for index in top3:
                    first, second = index[1], index[2]
                    pt1 = (int(joints[frame_index][first][0]),
                        int(joints[frame_index][first][1]))
                    pt2 = (int(joints[frame_index][second][0]),
                        int(joints[frame_index][second][1]))
                    if attention_id == 0 or attention_id == 3:
                        scaled_value_uint8 = int((1 - index[0]) * 255)
                    else:
                        scaled_value_uint8 = int((index[0]) * 255)

                    color_map = cv2.applyColorMap(
                        np.array([[scaled_value_uint8]]).astype(np.uint8), COLORMAP)
                    rgb_value = (int(color_map[0, 0, 0]), int(
                        color_map[0, 0, 1]), int(color_map[0, 0, 2]))
                    if No1:
                        cv2.line(frame, pt1, pt2, rgb_value, 2)
                        No1 = False
                    else:
                        cv2.line(frame, pt1, pt2, rgb_value, 1)
                concatenate_frame.append(frame)

            frame = np.concatenate(([i for i in concatenate_frame]), axis=1)
            frame_index += 1
            gif = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            gif = Image.fromarray(gif)
            output.append(gif)

            if (frame_index == num_frame):
                break

        output[0].save(output_path, save_all=True, append_images=output[1:], loop=0, disposal=2)
        cap.release()
        cv2.destroyAllWindows()


    def extract_x_y(bodys):
        joint = list()
        for body in bodys:
            keypoints_matrix = np.array(body["keypoints"]).reshape((17, 3))[:, :2]
            joint.append(keypoints_matrix)
        return joint


    def read_file(file_path, file_name, output_dir):
        with open(file_path, "rb") as f:
            labeldata = json.load(f)
        joints = extract_x_y(labeldata)
        draw(joints, output_dir, file_name)

    def read_dir(dir_path, output_dir):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.json'):
                    file_path = os.path.join(root, file)
                    file_name = os.path.basename(os.path.dirname(file_path))
                    file_name = file_name.split('e_')[1]
                    if (file_name in filenames):
                        read_file(file_path, file_name, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    read_dir(dir_path, output_dir)
