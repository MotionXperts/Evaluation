import pickle
import cv2, os
pkl_file = "/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl"

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

for item in data:
    if item['video_name'] == "467205989414993988_0" or item['video_name'] == "467205989414993988_1":
        print(item)
'''
pkl_file = "/home/andrewchen/Error_Localize/standard_features.pkl"
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

for item in data:
   print(item)
'''
file_path = "/home/c1l1mo/datasets/scripts/skating_pipeline/Loop/processed_videos/467205989414993988.mp4"
cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = 30 
frame_index = 0
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

save_dir = "saved_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

while frame_index < num_frames:
    ret, frame = cap.read()
    
    if not ret:
        break 
    frame_filename = os.path.join(save_dir, f"frame_{frame_index:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_index += 1

cap.release()
cv2.destroyAllWindows()
print(f"Finished saving {frame_index} frames to {save_dir}")