skeleton_body_part =    [   #[0, 1], [1, 2], [1, 3],  [0, 2],  [2, 4], 
                            [0,5], [0,6],  # face
                            [5, 7], [7, 9],                             # skeleton_left_hand
                            [6, 8], [8, 10],                            # skeleton_right_hand
                            [5, 6], [11,12],                            # hip and shoulder
                            [5, 11],[11, 13],[13, 15],                  # skeleton_left_leg
                            [6, 12],[12, 14],[14, 16]]                  # skeleton_right_leg

joint_graph_coordinate = [  [120, 20], # Nose
                            [115, 20], # R Eye
                            [125, 20], # L Eye
                            [110, 20], # R Ear
                            [130, 20], # L Ear
                            [140, 70], # R Shoulder
                            [100, 70], # L Shoulder
                            [160, 90], # R Elbow
                            [ 80, 90], # L Elbow
                            [165, 120],  # RWrist 
                            [ 75, 120], # LWrist
                            [140,140], # R hip
                            [100, 140], # L hip
                            [150,175], # R knee
                            [90, 175], # L knee
                            [150, 210], # R Ankle
                            [90, 210]] # L Ankle




# alpha_pose 1 will map to SMPL 12 
# alpha_pose 2 will map to SMPL 15
# alpha_pose 5 will map to SMPL 16
# alpha_pose 6 will map to SMPL 17
#                     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,11,12,13,14,15,16       # AlphaPose
AlphaPose_to_SMPL = [-1, 12, 15, -1, -1, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]      # SMPL

#                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21  # SMPL
SMPL_to_AlphaPose = [-1,11,12,-1,13,14,-1,15,16,-1,-1,-1, 1,-1,-1, 2, 5, 6, 7, 8, 9,10] # AlphaPose

SMPL_joints_name = ["pelvis",       "left hip",     "right hip",        "spine 1",      "left knee", 
                    "right knee",   "spine 2",      "left ankle",       "right ankle",  "spine 3", 
                    "left foot",    "right foot",   "neck",             "left collar",  "right collar", 
                    "head",         "left shoulder","right shoulder",   "left elbow",   "right elbow", 
                    "left wrist",   "right wrist"]

rank_color_list = [
    [[0, 0, 255], [73, 154, 255], [255, 194, 73]],
    [[0, 0, 255], [73, 154, 255], [255, 194, 73]],
    [[0, 0, 255], [73, 154, 255], [255, 194, 73]],
    [[0, 0, 255], [73, 154, 255], [255, 194, 73]],
]
# rank_color_list = [
#     [[255, 0, 0], [204, 102, 0], [153, 51, 0]],      # red
#     [[0, 0, 255], [80, 127, 255], [193, 182, 255]],  # blue
#     [[0, 255, 0], [34, 139, 34], [113, 179, 60]],    # green
#     [[255, 0, 255], [128, 0, 128], [221, 160, 221]]  # purple
# ]

backbone_color = [0, 255, 255]

combinations = [[i, j] for i in range(0, 22) for j in range(i+1, 22)]

AlphaPose_not_draw = [3,4]