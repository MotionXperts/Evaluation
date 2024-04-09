import os
import imageio
# coco keypoints: [x1,y1,v1,...,xk,yk,vk]       (k=17)
#     ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
#      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
def convert_mp4_to_gif(mp4_path, gif_path):
    # 使用 imageio 的 get_reader 函式讀取 MP4 檔案
    reader = imageio.get_reader(mp4_path)
    
    # 使用 imageio 的 get_writer 函式建立 GIF 檔案寫入器
    writer = imageio.get_writer(gif_path, duration=1/reader.get_meta_data()['fps'])
    
    try:
        # 遍歷 MP4 影片的每一幀，寫入 GIF 檔案
        for frame in reader:
            writer.append_data(frame)
    except Exception as e:
        print(f"Error converting MP4 to GIF: {e}")
    finally:
        # 關閉寫入器
        writer.close()

# 定義 MP4 檔案路徑和目標 GIF 檔案路徑
mp4_path = '/home/weihsin/datasets/Skating_Clip/467205307287470390_0/467205307287470390_0.mp4'
gif_path = '/home/weihsin/datasets/Skating_Clip/467205307287470390_0/467205307287470390_0.gif'

# 呼叫轉換函式
convert_mp4_to_gif(mp4_path, gif_path)