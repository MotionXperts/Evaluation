import os, pickle, torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

offset = 0.3
matplotlib.use('Agg')

def load_pkl(file_path) :
    print(file_path)
    with open(file_path, 'rb') as f :
        return pickle.load(f)

def normal_draw(joints_cam, file_name, gif_dir) :
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    
    cam_list = ["1", "2", "3"]
    color = ['gold', 'blue', 'red', 'lime']
    num_frame_min, frame_number= 100000, 0
    if(len(joints_cam) != 5) :
        # One of the four cameras is dedicated to capturing the name, indicating which athlete
        # and which perspective it represents.
        print("Do not have 4 camera view. The number of camera is ", len(joints_cam) - 1)
    for idx, i in enumerate(cam_list) :
            num_frame = len(joints_cam[i])
            if i != "2" and torch.equal(joints_cam[i], joints_cam["2"]) :
                print("The same error joints i is ", i)
            if (idx == 0) :
                frame_number = num_frame
                print("Correct : ", frame_number)
            else :
                if (frame_number != num_frame) :
                    print("Error the frame_number is ", frame_number , " and the num_frame is ", num_frame)
            num_frame_min = min(num_frame_min, num_frame)
            joints_cam[i] = joints_cam[i].reshape(num_frame, 22, 3)
            joints_cam[i][:, :, 1], joints_cam[i][:, :, 2] = -joints_cam[i][:, :, 1], -joints_cam[i][:, :, 2]
    def animate(frame) :
        ax.clear()

        for idx, i in enumerate(cam_list) :
            try :
                joints = joints_cam[i]
            except :
                print("Error joint i is ", i)
                continue
            skeleton = joints[frame]
            ax.set_xlabel('x axis')
            ax.set_zlabel('y axis')
            ax.set_ylabel('z axis')
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(-0.4, 0.4)
            ax.set_zlim(-0.4, 0.4)
            # ax.view_init(elev = 0, azim = 90)
            skeleton_bone = [[15, 12],[12, 9],[9, 6],[6, 3],[3, 0]]
            skeleton_left_leg = [[0, 1],[1, 4],[4, 7],[7, 10]]
            skeleton_right_leg = [[0, 2],[2, 5],[5, 8],[8, 11]]
            skeleton_left_hand = [[9, 13],[13, 16],[16, 18],[18, 20]]
            skeleton_right_hand = [[9, 14],[14, 17],[17, 19],[19, 21]]
            
            for index in skeleton_bone : 
                first, second = index[0], index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]],
                        [skeleton[first][1], skeleton[second][1]],
                        [skeleton[first][2], skeleton[second][2]], color[idx], linewidth = 1.0) # gold
            
            for index in skeleton_left_leg :
                first, second = index[0], index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]],
                        [skeleton[first][1], skeleton[second][1]],
                        [skeleton[first][2], skeleton[second][2]], color[idx], linewidth = 1.0) # cyan
        
            for index in skeleton_right_leg :
                first, second = index[0], index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]],
                        [skeleton[first][1], skeleton[second][1]],
                        [skeleton[first][2], skeleton[second][2]], color[idx], linewidth = 1.0) # fuchsia
            
            for index in skeleton_left_hand :
                first, second = index[0], index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]],
                        [skeleton[first][1], skeleton[second][1]],
                        [skeleton[first][2], skeleton[second][2]], color[idx], linewidth = 1.0) # lime
    
            for index in skeleton_right_hand :
                first, second = index[0], index[1]
                ax.plot([skeleton[first][0], skeleton[second][0]],
                        [skeleton[first][1], skeleton[second][1]],
                        [skeleton[first][2], skeleton[second][2]], color[idx], linewidth = 1.0) # red

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

    ani = animation.FuncAnimation(fig, animate, frames = num_frame_min, repeat = True)
    gif = file_name + '.gif'
    gif_path = os.path.join(gif_dir, gif)
    print("gif_path", gif_path)
    ani.save(gif_path, fps = 100)

# Use command
# python camera_boxing_read.py
if __name__ == "__main__" :
    # train boxing path
    # train_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_GT_train_aggregate.pkl"
    # test boxing path
    # test_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_GT_test_aggregate.pkl"
    # train_dataset = load_pkl(train_path)
    # test_dataset = load_pkl(test_path)

    data_dict = {}
    '''
    for data in train_dataset :
        video_name = data['video_name']
        prefix = video_name.split('_cam')[0]
        postfix = video_name.split('_cam')[1]
        if prefix not in data_dict :
            data_dict[prefix] = {}
        data_dict[prefix][postfix] = data["features"]
        data_dict[prefix]["name"] = prefix

    for data in test_dataset :
        video_name = data['video_name']
        prefix = video_name.split('_cam')[0]
        postfix = video_name.split('_cam')[1]
        if prefix not in data_dict :
            data_dict[prefix] = {}
        data_dict[prefix][postfix] = data["features"]
        data_dict[prefix]["name"] = prefix
    '''
    all_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_all_cam_sorted.pkl"
    all_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing.pkl"
    gif_dir = "./Boxing_camera_visualize"
    all_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_all_cam_aligned.pkl"
    gif_dir = "./Boxing_camera_visualize2"
    test_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_test.pkl"
    train_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_train.pkl"
    gif_dir = "./Boxing_camera_visualize3"
    if not os.path.exists(gif_dir) :
        os.makedirs(gif_dir)
    test_dataset = load_pkl(test_path)
    train_dataset = load_pkl(train_path)
    all_dataset = test_dataset + train_dataset
    for data in all_dataset :
        video_name = data['video_name']
        prefix = video_name.split('_cam')[0]
        postfix = video_name.split('_cam')[1]
        if (postfix != "1" and postfix != "2" and postfix != "3" and postfix != "4") :
            print("Error", postfix)
        if prefix not in data_dict :
            data_dict[prefix] = {}
        data_dict[prefix][postfix] = data["features"]
        data_dict[prefix]["name"] = prefix

    for data in data_dict :
        normal_draw(data_dict[data], data_dict[data]["name"], gif_dir)
        print(data_dict[data]["name"]," finish.")