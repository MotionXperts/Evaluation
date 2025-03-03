import os, json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoFileClip
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from PIL import Image, ImageSequence
import argparse
from drawutils.rank_color import skeleton_body_part, joint_graph_coordinate, AlphaPose_to_SMPL, SMPL_to_AlphaPose
from drawutils.rank_color import SMPL_joints_name, rank_color_list, backbone_color, combinations,AlphaPose_not_draw
from drawutils.read_file import read_file, read_config, findfilename
from drawutils.Top3 import readattention, Top3Node, Top3Link, readframe, readtext
matplotlib.use('Agg')

# not include 0 3 6 9 10 11 13 14
not_include = []
for i in range(0, 22):
    if i not in AlphaPose_to_SMPL:
        not_include.append(i)

def read_start_frame(file_name):
    base_paths = {
    "Axel_com": "/home/weihsin/datasets/Axel_com",
    "Axel": "/home/weihsin/datasets/Axel",
    "Flip_com": "/home/weihsin/datasets/Flip_com",
    "Flip": "/home/weihsin/datasets/Flip",
    "Loop_com": "/home/weihsin/datasets/Loop_com",
    "Loop": "/home/weihsin/datasets/Loop",
    "Lutz_com": "/home/weihsin/datasets/Lutz_com",
    "Lutz": "/home/weihsin/datasets/Lutz",
    "Salchow_com": "/home/weihsin/datasets/Salchow_com",
    "Salchow": "/home/weihsin/datasets/Salchow",
    "Sit_spin": "/home/weihsin/datasets/Sit_spin",
    "Toe-Loop": "/home/weihsin/datasets/Toe-Loop"
}
    action_type = file_name.split('_')[0]
    number = file_name.split('_')[1].split('.mp4')[0]
    # print(number)
    json_file = "U38e032ad764583bb60d61dc37019433b.json"
    for key, base_path in base_paths.items():
        action_path = os.path.join(base_path, action_type)
        if os.path.exists(action_path):
            target_file = os.path.join(action_path, json_file)
            print("Checking:", target_file)
            if not os.path.exists(target_file):
                print(f"File {target_file} does not exist.")
                return -1, -1

            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # print("item[id]", item["id"])
                    if int(item["id"]) == int(number):
                        # print("success")
                        start_time, end_time = map(float, item["timestamp"].split(" - "))
                        # print("start_time",start_time)
                        # print("end_time",end_time)
                        return int(start_time * 30), int(end_time * 30)
            return -1, -1

    print("No matching path found.")
    return -1, -1
    
def draw_body_graph():
    # color all frame to black
    frame = np.zeros((224, 224, 3), np.uint8)
    for i in range(0, 17):
        if i in AlphaPose_not_draw:
            continue
        if i == 0:
            cv2.circle(frame, (int(joint_graph_coordinate[i][0]), int(joint_graph_coordinate[i][1])), 20, backbone_color, -1)
        else:
            cv2.circle(frame, (int(joint_graph_coordinate[i][0]), int(joint_graph_coordinate[i][1])), 2, backbone_color, -1)
    neck_coordinate = ( int((joint_graph_coordinate[5][0] + joint_graph_coordinate[6][0])/2),
                        int((joint_graph_coordinate[5][1] + joint_graph_coordinate[6][1])/2))
    nose_coordinate = (int(joint_graph_coordinate[0][0]), int(joint_graph_coordinate[0][1]))
    cv2.line(frame, nose_coordinate, neck_coordinate, backbone_color, 3)
    for link_pair in skeleton_body_part:
        A,B = link_pair[0], link_pair[1]
        if A == 0 or A == 1 or A == 2 or A == 3 or B == 0 or B == 1 or B == 2 or B == 3 : continue
        pt1, pt2 = (int(joint_graph_coordinate[A][0]), int(joint_graph_coordinate[A][1])) , (int(joint_graph_coordinate[B][0]), int(joint_graph_coordinate[B][1]))
        cv2.line(frame, pt1, pt2, backbone_color, 3)
    return frame

def draw(joints, output_path, file_name, video_path_dir, attention_node_path, attention_matrix_path, index_dict_path, max_index_path, results_pth):

    video_path = video_path_dir +'/alpha_pose_' + file_name.split('_')[0] + "/" + file_name.split('_')[0] + ".mp4"
    output_path_origin = output_path
    output_gif_path = os.path.join(output_path, file_name + '.gif')

    num_frame = len(joints)

    color_node, attention_matrix_new = readattention(file_name, num_frame, 0, attention_node_path, attention_matrix_path)
    # print("color node frame",len(color_node))
    start_frame, end_frame = read_start_frame(file_name)
    if (end_frame - start_frame > len(color_node)):
        end_frame = start_frame + len(color_node)
    if (end_frame > num_frame):
        end_frame = num_frame
    if (start_frame) == -1 :
        print("Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        start_frame = 0
        end_frame = num_frame - 1
        return
    frame_index = start_frame - 1
    # print("start_frame",start_frame)
    # print("end_fream",end_frame)
    output, gif = [], []
    key_frame, start_end_frame = readframe(file_name, index_dict_path, max_index_path)
    # print("key frame", key_frame)
    # print("start_end_frame",start_end_frame)
    # print("num_frame",num_frame)
    text = readtext(file_name, results_pth)

    ''' Run all frame of a video '''
    while (frame_index != num_frame):
        frame_index += 1
        image_path = os.path.join(video_path_dir +'/alpha_pose_' + file_name.split('_')[0] , "vis")
        image_path = os.path.join(image_path, str(frame_index) + ".jpg")
        if not os.path.exists(image_path):
            continue
        frame_c = cv2.imread(image_path)

        if frame_index >= end_frame :
            continue
        ''' Four attention matrix '''
        attention_list = []
        for attention_id in [0,1,2,3]:
            skeleton_graph_frame = draw_body_graph()
            skeleton_time_frame = draw_body_graph()
            rank_color = rank_color_list[attention_id]
            color_node, attention_matrix_new = readattention(file_name, num_frame, attention_id, attention_node_path, attention_matrix_path)
            frame = frame_c.copy()
            time_frame = frame_c.copy()
            ori_frame = frame_c.copy()
            COLORMAP = cv2.COLORMAP_SPRING
            if (True) :
                '''
                Attention Node
                '''
                # Attention node of according frame_index
                error_frame = frame_index - start_frame #start_end_frame["trimmed_start"]
                # print(error_frame)
                # print(start_frame)
                colornodes = np.array(color_node[error_frame])
                # Sort the SMPL node in descending order
                indexmax = np.argsort(-colornodes)

                for i in range(0, len(indexmax)):
                    if (indexmax[i] in not_include):
                        colornodes[indexmax[i]] = -1

                # Sort the SMPL node that can mapped to Alphapose in descending order
                indexmax = np.argsort(-colornodes)

                x_coordinates, y_coordinates = joints[frame_index][:,0], joints[frame_index][:,1]

                '''
                Attention Link
                '''
                all_link, all_graph_link = [], [] # Top 3 link
                body_link = [] # All link

                for SMPL_pair in combinations: # Top 3 link
                    SMPL_A,SMPL_B = SMPL_pair[0], SMPL_pair[1]
                    if(SMPL_to_AlphaPose[SMPL_A] != -1 and SMPL_to_AlphaPose[SMPL_B] != -1):
                        link = {}
                        link['SMPL_pair']           = SMPL_pair
                        link['AlphaPose_pair']      = [SMPL_to_AlphaPose[SMPL_A], SMPL_to_AlphaPose[SMPL_B]]
                        link['value']               = attention_matrix_new[SMPL_A][SMPL_B] + attention_matrix_new[SMPL_B][SMPL_A]
                        link['A_x'], link['A_y']    = x_coordinates[SMPL_to_AlphaPose[SMPL_A]], y_coordinates[SMPL_to_AlphaPose[SMPL_A]]
                        link['B_x'], link['B_y']    = x_coordinates[SMPL_to_AlphaPose[SMPL_B]], y_coordinates[SMPL_to_AlphaPose[SMPL_B]] 
                        all_link.append(link)

                        graph_link = {}
                        graph_link['SMPL_pair']           = SMPL_pair
                        graph_link['value'] = link['value']
                        graph_link['A_x'], graph_link['A_y'] = joint_graph_coordinate[link['AlphaPose_pair'][0]][0], joint_graph_coordinate[link['AlphaPose_pair'][0]][1]
                        graph_link['B_x'], graph_link['B_y'] = joint_graph_coordinate[link['AlphaPose_pair'][1]][0], joint_graph_coordinate[link['AlphaPose_pair'][1]][1]                                                     
                        all_graph_link.append(graph_link)

                all_link.sort(key=lambda x: x['value'], reverse=True)
                Top3Link = all_link[:3]
                all_graph_link.sort(key=lambda x: x['value'], reverse=True)
                Top3GraphLink = all_graph_link[:3]
                '''
                All Link
                '''
                for link_pair in skeleton_body_part: # All link
                    A,B = link_pair[0], link_pair[1]
                    if A in AlphaPose_not_draw or B in AlphaPose_not_draw : continue
                    if(SMPL_to_AlphaPose[SMPL_A] != -1 and SMPL_to_AlphaPose[SMPL_B] != -1):
                        link = {}
                        link['A_x'], link['A_y'], link['B_x'], link['B_y'] = x_coordinates[A], y_coordinates[A], x_coordinates[B], y_coordinates[B]
                        body_link.append(link)

                for link in body_link:
                    pt1, pt2 = (int(link['A_x']), int(link['A_y'])) , (int(link['B_x']), int(link['B_y']))
                    cv2.line(frame, pt1, pt2, backbone_color, 2)

                for i in range(0, 3):
                    Top3Link[i]['rank'] = i
                    Top3GraphLink[i]['rank'] = i

                for link in Top3Link:
                    pt1, pt2 = (int(link['A_x']), int(link['A_y'])) , (int(link['B_x']), int(link['B_y']))
                    '''
                    Setting color weight according by attention id
                    Color map will be reberse the color weight (ex COLORMAP_SPRING & COLORMAP_AUTUMN)
                    '''
                    # color_map = cv2.applyColorMap(np.array([[0 + 127 * link['rank']]]).astype(np.uint8), COLORMAP)
                    # rgb_value = (int(color_map[0, 0, 0]), int(color_map[0, 0, 1]), int(color_map[0, 0, 2]))
                    cv2.line(frame, pt1, pt2, rank_color[link['rank']], 10 - 3*link['rank'])
                    if((key_frame[link['SMPL_pair'][0]] - 3 < error_frame and error_frame < key_frame[link['SMPL_pair'][0]] + 3) or
                       (key_frame[link['SMPL_pair'][1]] - 3 < error_frame and error_frame < key_frame[link['SMPL_pair'][1]] + 3)):
                        cv2.line(time_frame, pt1, pt2, rank_color[link['rank']], 10 - 3*link['rank'])
                for link in Top3GraphLink:
                    pt1, pt2 = (int(link['A_x']), int(link['A_y'])) , (int(link['B_x']), int(link['B_y']))
                    cv2.line(skeleton_graph_frame, pt1, pt2, rank_color[link['rank']], 10 - 3*link['rank'])
                    if((key_frame[link['SMPL_pair'][0]] - 3 < error_frame and error_frame < key_frame[link['SMPL_pair'][0]] + 3) or
                       (key_frame[link['SMPL_pair'][1]] - 3 < error_frame and error_frame < key_frame[link['SMPL_pair'][1]] + 3)):
                        cv2.line(skeleton_time_frame, pt1, pt2, rank_color[link['rank']], 10 - 3*link['rank'])

                '''
                Attention Node
                '''

                Top3Node = []
                for i in range(0, 3):
                    node = {}
                    node['SMPL_index']      = indexmax[i]
                    node['AlphaPose_index'] = SMPL_to_AlphaPose[indexmax[i]]
                    node['value']           = colornodes[indexmax[i]]
                    node['rank']            = i
                    node['x'], node['y']    = x_coordinates[node['AlphaPose_index']],  y_coordinates[node['AlphaPose_index']]
                    Top3Node.append(node)

                for node in Top3Node:
                    '''
                    Setting color weight according by attention id
                    Color map will be reberse the color weight (ex COLORMAP_SPRING & COLORMAP_AUTUMN)
                    '''
                    # color_map = cv2.applyColorMap(np.array([[0 + 127 * node['rank']]]).astype(np.uint8), COLORMAP)
                    # rgb_value = (int(color_map[0, 0, 0]), int(color_map[0, 0, 1]), int(color_map[0, 0, 2]))
                    cv2.circle(frame, (int(node['x']), int(node['y'])), int(14 - node['rank']*3), rank_color[node['rank']], -1)
                    if(key_frame[node['SMPL_index']] - 3 < error_frame and error_frame < key_frame[node['SMPL_index']] + 3) :
                        cv2.circle(time_frame, (int(node['x']), int(node['y'])), int(14 - node['rank']*3), rank_color[node['rank']], -1)
                    cv2.circle(skeleton_graph_frame, (int(joint_graph_coordinate[node['AlphaPose_index']][0]), int(joint_graph_coordinate[node['AlphaPose_index']][1])), int(14 - node['rank']*3), rank_color[node['rank']], -1)
                    if(key_frame[node['SMPL_index']] - 3 < error_frame and error_frame < key_frame[node['SMPL_index']] + 3) :
                        cv2.circle(skeleton_time_frame, (int(node['x']), int(node['y'])), int(14 - node['rank']*3), rank_color[node['rank']], -1)

                '''
                All Node

                allNode = []
                for i in range(0, 17):
                    node = {}
                    node['x'], node['y'] = x_coordinates[i], y_coordinates[i]
                    allNode.append(node)
                for node in allNode:
                    cv2.circle(frame, (int(node['x']), int(node['y'])), 2, backbone_color, -1)
                '''

            # concatenate_frame.append(frame)
            skeleton_graph_frame = cv2.cvtColor(skeleton_graph_frame, cv2.COLOR_BGR2RGBA)
            skeleton_time_frame = cv2.cvtColor(skeleton_time_frame, cv2.COLOR_BGR2RGBA)
            time_frame =  cv2.cvtColor(time_frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            ori_frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGBA)
            frames = [ori_frame, frame, skeleton_graph_frame, time_frame, skeleton_time_frame]
            # attention_graph = np.concatenate([ori_frame, frame, skeleton_graph_frame, time_frame, skeleton_time_frame], axis=1)
            titles = ["Original Video", "Attention Skeleton", "Attention Skeleton Graph", "Error Skeleton", "Error Skeleton Graph"]
            title_height  = 30
            annotated_frames = []
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 0, 0, 255)  # black text
            background_color = (255, 255, 255, 255)  # white background
            thickness = 1
            for frame, title in zip(frames, titles):
                # create a white background as the title bar
                title_row = np.ones((title_height, frame.shape[1], 4), dtype=np.uint8) * 255  # white background

                # calculate the position of text
                text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
                text_x = (title_row.shape[1] - text_size[0]) // 2   # horizontally centered
                text_y = (title_height + text_size[1]) // 2         # vertically centered

                # draw text on the title
                cv2.putText(title_row, title, (text_x, text_y), font, font_scale, text_color, thickness)

                # vertical concatenate the title bar and image
                annotated_frame = np.concatenate([title_row, frame], axis=0)
                annotated_frames.append(annotated_frame)

            # Combine all the processed images horizontally into a single attention graph.
            attention_graph = np.concatenate(annotated_frames, axis=1)
            # For Instruction
            font_scale = 1.0
            thickness = 2
            text_height = 200
            # white background
            text_row = np.ones((text_height, attention_graph.shape[1], 4), dtype=np.uint8) * 255

            # Calculate the maximum width of each row (i.e., the background width).
            max_width = text_row.shape[1]

            # split text into multiple line
            words = text.split(" ")
            lines = []
            current_line = ""
            for word in words:
                # Try adding a word to the current row.
                test_line = current_line + (" " if current_line else "") + word
                text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
                if text_size[0] <= max_width:
                    current_line = test_line
                else:
                    # When the current row is full, move to the next row.
                    lines.append(current_line)
                    current_line = word

            # Add the last row.
            if current_line:
                lines.append(current_line)

            line_height = 50        # The vertical spacing between each line of text
            y_offset = line_height  # initial offset
            for line in lines:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = (max_width - text_size[0]) // 2    # horizontally centered
                text_y = y_offset                           # vertically centered
                cv2.putText(text_row, line, (text_x, text_y), font, font_scale, text_color, thickness)
                y_offset += line_height

            final_result = np.concatenate([text_row, attention_graph], axis=0)

            attention_graph = Image.fromarray(final_result, 'RGBA')
            attention_list.append(attention_graph)

        output.append(attention_list)
        # print("file_name",file_name)

    # Take msec as unit, fps is N. Duration means the time of each frame. Thus duration = 1000 / N.
    fps = 30
            
    if not os.path.exists(output_path_origin + '/' + file_name):
        os.makedirs(output_path_origin + '/' + file_name)

    # save all single frame to png the path folder
    # output " [number of frame] * [number of attention matrix] "
    new_output = []
    for i in range(0,4) :
        for j in range(len(output)):
            new_output.append(output[j][i])
    # print("new_output",len(new_output))
    for i in range(len(new_output)):
        new_output[i].save(output_path_origin + '/' + file_name + '/' + file_name + '_'+ str(i+start_frame) + '.png')
    new_output[0].save(output_gif_path, save_all=True, append_images=new_output[1:], loop=0, disposal=2, duration=1000/fps)
    #ori_frame.save()

    cv2.destroyAllWindows()

# Use command
# python draw_skeleton2D.py /home/weihsin/projects/Evaluation/config_file/finetuneAttention.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read configuration file and run setup.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    attention_node, attention_matrix, epoch_num, output_dir, video_dir, golden_file, index_dict, max_index, results = read_config(args.config_path)

    # When running the error segment setting, it is must to check the file name in the attention_node file
    All_filenames = findfilename(attention_node)
    # print(len(All_filenames))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    golden_file_list = []
    if golden_file is not None:
        with open(golden_file) as f:
            # golden_file is txt file
            golden_file = open(golden_file, 'r').read().splitlines()
            for line in golden_file:
                golden_file_list.append(line)

        for filename in All_filenames:
            if filename not in golden_file_list:
                All_filenames.remove(filename)
    if golden_file is not None:
        All_filenames = golden_file_list
    print(All_filenames)

    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(os.path.dirname(file_path))
                file_name = file_name.split('e_')[1]
                if file_name  in All_filenames :
                    print("Processing", file_name)
                    joints = read_file(file_path)
                    draw(joints, output_dir, file_name, video_dir, attention_node, attention_matrix, index_dict, max_index, results)

                if file_name + '_0' in All_filenames :
                    file_name0  = file_name + '_0'
                    print("Processing", file_name0)
                    joints = read_file(file_path)
                    draw(joints, output_dir, file_name0, video_dir,attention_node ,attention_matrix, index_dict, max_index, results)

                if file_name + '_1' in All_filenames :
                    file_name1  = file_name + '_1'
                    print("Processing", file_name1)
                    joints = read_file(file_path)
                    draw(joints, output_dir, file_name1, video_dir,attention_node ,attention_matrix, index_dict, max_index, results)

                if file_name + '_2' in All_filenames :
                    file_name2  = file_name + '_2'
                    print("Processing", file_name2)
                    joints = read_file(file_path)
                    draw(joints, output_dir, file_name2, video_dir,attention_node ,attention_matrix, index_dict, max_index, results)
                    # Top3Node(file_name, attention_node, output_dir)
                    # Top3Link(file_name, attention_matrix, output_dir)