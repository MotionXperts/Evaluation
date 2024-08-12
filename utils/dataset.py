import pickle
import shutil
import os

pkl_file_path = '/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl'
target_base_dir = '/home/weihsin/datasets/Skating_Dataset0811'

with open(pkl_file_path, 'rb') as f:
    data_entries = pickle.load(f)

for entry in data_entries:
    video_path = entry['video_file']
    video_name = os.path.basename(video_path).split('.')[0]
    alpha_pose_filename = "alpha_pose_" + video_name + "/" + video_name + ".mp4"

    filename = os.path.basename(video_path)
    target_dir = os.path.join(target_base_dir, alpha_pose_filename)
    shutil.copy(video_path, target_dir)