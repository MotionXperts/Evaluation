import os, json, pickle, numpy as np, torch
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
offset = 0.3
matplotlib.use('Agg')
from smplx import SMPL

def load_joints(data):
    poses = torch.tensor(data["poses"][0]).float() # shape : (69,)
    poses = poses.unsqueeze(0)                     # shape : (1, 69)
    Rh = torch.tensor(data['Rh'][0], dtype = torch.float32).unsqueeze(0) # shape : (1, 3)
    Th = torch.tensor(data['Th'][0], dtype = torch.float32).unsqueeze(0) # shape : (1, 3)

    shapes = torch.tensor(data['shapes'][0], dtype = torch.float32).unsqueeze(0) # shape : (1, 10)

    model = SMPL(model_path = './basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl', gender = 'neutral').eval()
    output = model(
        global_orient = Rh.unsqueeze(0),  # shape : (1, 1, 3)
        body_pose = poses.view(1, 23, 3), # shape : (1, 23, 3)
        betas = shapes,                   # shape : (1, 10)
        transl = Th                       # shape : (1, 3)
    )

    joints_3d = output.joints.detach().cpu().numpy()

    return joints_3d[0]

def normal_draw(joints, file_name, gif_dir) :
    fig = plt.figure(figsize = (10, 10))
    ax = plt.subplot(projection = '3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    color = ['gold', 'blue', 'red', 'lime']

    num_frame = joints.shape[0]
    def animate(frame) :
        ax.clear()

        skeleton = joints[frame]
        ax.set_xlabel('x axis')
        ax.set_zlabel('y axis')
        ax.set_ylabel('z axis')

        skeleton_bone = [[15, 12],[12, 9],[9, 6],[6, 3],[3, 0]]
        skeleton_left_leg = [[0, 1],[1, 4],[4, 7],[7, 10]]
        skeleton_right_leg = [[0, 2],[2, 5],[5, 8],[8, 11]]
        skeleton_left_hand = [[9, 13],[13, 16],[16, 18],[18, 20]]
        skeleton_right_hand = [[9, 14],[14, 17],[17, 19],[19, 21]]
            
        for index in skeleton_bone : 
            first, second = index[0], index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]],
                    [skeleton[first][1], skeleton[second][1]],
                    [skeleton[first][2], skeleton[second][2]], color[0], linewidth = 1.0) # gold
            
        for index in skeleton_left_leg :
            first, second = index[0], index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]],
                    [skeleton[first][1], skeleton[second][1]],
                    [skeleton[first][2], skeleton[second][2]], color[0], linewidth = 1.0) # cyan
        
        for index in skeleton_right_leg :
            first, second = index[0], index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]],
                    [skeleton[first][1], skeleton[second][1]],
                    [skeleton[first][2], skeleton[second][2]], color[0], linewidth = 1.0) # fuchsia
            
        for index in skeleton_left_hand :
            first, second = index[0], index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]],
                    [skeleton[first][1], skeleton[second][1]],
                    [skeleton[first][2], skeleton[second][2]], color[0], linewidth = 1.0) # lime

        for index in skeleton_right_hand :
            first, second = index[0], index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]],
                    [skeleton[first][1], skeleton[second][1]],
                    [skeleton[first][2], skeleton[second][2]], color[0], linewidth = 1.0) # red

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

    ani = animation.FuncAnimation(fig, animate, frames = num_frame, repeat = True)
    gif = file_name + '.gif'
    gif_path = os.path.join(gif_dir, gif)
    print("gif_path", gif_path)
    ani.save(gif_path, fps = 100)

# Use command
# python camera_newboxing_read.py
def load_pkl(file_path) :
    print(file_path)
    with open(file_path, 'rb') as f :
        return pickle.load(f)

if __name__ == "__main__" :
    '''
    test_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_test.pkl"
    test_dataset = load_pkl(test_path)
    # print its key
    print(test_dataset[0].keys())
    boxing_dataset_path = "/home/weihsin/datasets/BoxingDatasetPkl"
    save_path = os.path.join(boxing_dataset_path, "boxing_2025-03-18.pkl")
    new_dataset = load_pkl(save_path)
    print("new_dataset[0]", new_dataset[0].keys())
    '''
    data_dict = []
    boxing_dataset_path = "/home/weihsin/datasets/BoxingDatasetPkl"
    dir_path = os.path.join(boxing_dataset_path, "boxing_2025-03-18_results")
    gif_dir = "./Boxing_new_visualize"
    for folder_num in range(0, 36) :
        folder_name = f"{folder_num:06d}"
        keypoints_dir = os.path.join(dir_path, folder_name, "smpl")
        
        if not os.path.isdir(keypoints_dir) :
            print(f"Skipping missing folder : {keypoints_dir}")
            continue

        json_files = sorted([f for f in os.listdir(keypoints_dir) if f.endswith(".json")])

        joints_list = []

        for file_name in tqdm(json_files, desc = f"Processing {folder_name}"):
            file_path = os.path.join(keypoints_dir, file_name)

            with open(file_path, "r") as f:
                data = json.load(f)

            keypoints = load_joints(data[0])
            joints_list.append(keypoints)

        joints = np.array(joints_list)  # shape : (num_frames, 22, 3)
        
        data = {"features" : joints.tolist(),
                "video_name" : folder_name}

        data_dict.append(data)
        normal_draw(joints, folder_name, gif_dir)

    save_path = os.path.join(boxing_dataset_path, "new_boxing_dataset_20250318.pkl")
    with open(save_path, "wb") as f :
        pickle.dump(data_dict, f)
    print(f"Data saved to {save_path}")
    
