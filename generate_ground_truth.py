import pickle as pkl
import numpy as np
import os
import json
path = './datasets/FigureSkate/HumanML3D_g/global_human_test.pkl'
path = './datasets/Loop/test_Loop.pkl'
path1 = './datasets/VQA/test_local.pkl'
path2 = './datasets/VQA/train_local.pkl'

filepath = 'humalMLgroundtruth.json'

setting = "Skating"

if setting == "Boxing" :
    ## Boxing
    # train boxing path 
    train_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_GT_train_aggregate.pkl"
    # test boxing path
    test_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_GT_test_aggregate.pkl"

    with open(train_path, 'rb') as f:
        data1 = pkl.load(f)
    with open(test_path, 'rb') as f:
        data2 = pkl.load(f)
        
    filepath = 'boxing_GT.json'

if setting == "Skating" :
    ## Skating
    # train boxing path 
    train_path = "/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_GT_train/aggregate.pkl"
    # test boxing path
    test_path = "/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl"

    with open(train_path, 'rb') as f:
        data1 = pkl.load(f)
    with open(test_path, 'rb') as f:
        data2 = pkl.load(f)
        
    filepath = 'Skating_GT.json'

dictitory = {}
if setting == "Boxing" :
    for i in range(len(data1)):
        print(data1[i]['video_name'])
        print(data1[i]['labels'])
        video_name = data1[i]['video_name']
        labels   = data1[i]['labels']
        dictitory[video_name] = labels

    for i in range(len(data2)):
        print(data2[i]['video_name'])
        print(data2[i]['labels'])
        video_name = data2[i]['video_name']
        labels   = data2[i]['labels']
        dictitory[video_name] = labels

if setting == "Skating" :
    for i in range(len(data1)):

        video_name = data1[i]['video_name']
        labels = data1[i]['augmented_labels']
        labels.append(data1[i]['revised_label'])
        dictitory[video_name] = labels

    for i in range(len(data2)):

        video_name = data2[i]['video_name']
        labels = data2[i]['augmented_labels']
        labels.append(data2[i]['revised_label'])
        dictitory[video_name] = labels

with open(filepath, 'w') as f:
    json.dump 
    json.dump(dictitory, f, indent=4)