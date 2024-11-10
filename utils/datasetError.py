import pickle
import shutil
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

pkl_file_path = '/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl'
pkl_file_path = '/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_segment_test/aggregate.pkl'
target_base_dir = '/home/weihsin/datasets/Skating_Dataset0815'
pkl_file_path = '/home/andrewchen/Error_Localize/boxing_aggregate_error_segment.pkl'
target_base_dir = '/home/weihsin/datasets/Boxing_Dataset_error'
def clip_video_by_frames(input_file, output_file, start_frame, end_frame, fps):
    # Calculate the start and end times in seconds
    start_time = start_frame / fps
    end_time = end_frame / fps
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

with open(pkl_file_path, 'rb') as f:
    data_entries = pickle.load(f)

for entry in data_entries:
    #video_name = entry['original_video_name']
    #video_path = entry['original_video_file']
    video_name = entry['video_name']
    video_path = entry['video_file']
    video_name_json = entry['video_name']

    video_dir = "alpha_pose_" + video_name_json + "/"
    if not os.path.exists(os.path.join(target_base_dir, video_dir)):
        os.makedirs(os.path.join(target_base_dir, video_dir))

    original_video_filename = "alpha_pose_" + video_name_json + "/" + video_name_json + "_ori.mp4"
    alpha_pose_filename = "alpha_pose_" + video_name_json + "/" + video_name_json + ".mp4"

    trimmed_start = entry['trimmed_start']
    if not entry['standard_longer']:
        start_frame = entry['start_frame'] + trimmed_start
        end_frame = entry['end_frame'] + trimmed_start
        # entry['std_start_frame'] = 0
        # entry['std_end_frame'] = end_frame - start_frame

        # The following code is same as the line 108 109 in repository MotionExpert/dataloader.py
        feature_start_frame = start_frame + int(entry['error_start_frame'])
        feature_end_frame = start_frame + (int(entry['error_end_frame'])-1)
    else:
        entry['std_start_frame'] = entry['start_frame']
        entry['std_end_frame'] = entry['end_frame'] - 1
        start_frame = trimmed_start
        end_frame = trimmed_start + (entry['std_end_frame'] - entry['std_start_frame'])

        # The following code is same as the line 113 114 in repository MotionExpert/dataloader.py
        feature_start_frame = int(entry['trimmed_start']) + int(entry['error_start_frame'])
        feature_end_frame = int(entry['trimmed_start']) + (int(entry['error_end_frame'])-1)

    print("original_video_filename",original_video_filename)
    original_file_path = os.path.join(target_base_dir, original_video_filename)
    shutil.copy(video_path, original_file_path)

    target_file_path = os.path.join(target_base_dir, alpha_pose_filename)

    clip_video_by_frames(original_file_path, target_file_path, feature_start_frame, feature_end_frame,30)

    filename = os.path.basename(video_path)
    print("Finished processing", filename)