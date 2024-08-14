import argparse
import yaml
import pickle
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = config['output_dir']
    return output_dir

def analyze(target_path):
    with open(target_path) as f:
        data_entries = json.load(f)

    matrix_calculate0 = np.zeros(3)
    matrix_calculate1 = np.zeros(3)
    matrix_calculate2 = np.zeros(3)
    matrix_calculate3 = np.zeros(3)
    node_calculate = np.zeros(3)
    data_number = len(data_entries)
    for data in data_entries:
        items = [item.strip() for item in data['response'].split(',')]

        filtered_items = [item for item in items if not re.search(r'\b(Matrix|Node)\b', item)]
        unique_items = list(set(filtered_items))

        matrix_name = ['0','1','2','3']
        rank_name = ['rank_1', 'rank_2', 'rank_3']
        for item in unique_items:
            if(item == 'spine') : 
                continue
            for i in range(len(matrix_name)):
                for j in range(len(rank_name)):
                    if item in data['matrix'][matrix_name[i]][rank_name[j]][0] or item in data['matrix'][matrix_name[i]][rank_name[j]][1]:
                        if i == 0:
                            matrix_calculate0[j] += 1
                        if i == 1:
                            matrix_calculate1[j] += 1
                        if i == 2:
                            matrix_calculate2[j] += 1
                        if i == 3:
                            matrix_calculate3[j] += 1
                        
            for j in range(len(rank_name)):
                if item in data['node'][rank_name[j]] :
                    node_calculate[j] += 1
       
    return matrix_calculate0, matrix_calculate1, matrix_calculate2, matrix_calculate3, node_calculate, data_number

def draw_graph(matrix_calculate0, matrix_calculate1, matrix_calculate2, matrix_calculate3, node_calculate, output_dir, data_number):
    output_path = output_dir + '/rank_withoutspine.png'

    values = [matrix_calculate0[0], matrix_calculate0[1], matrix_calculate0[2],
              matrix_calculate1[0], matrix_calculate1[1], matrix_calculate1[2],
              matrix_calculate2[0], matrix_calculate2[1], matrix_calculate2[2],
              matrix_calculate3[0], matrix_calculate3[1], matrix_calculate3[2],
              node_calculate[0], node_calculate[1], node_calculate[2]]
    for index in range(0,len(values)) :
        values[index] = (values[index] / data_number) * 100

    color1 = 'hotpink'
    color2 = 'lightsalmon'
    color3 = 'gold'
    colors = [color1, color2, color3,
              color1, color2, color3,
              color1, color2, color3,
              color1, color2, color3,
              color1, color2, color3]

    n = 15
    x = np.arange(n)

    plt.legend(handles=[
        plt.Rectangle((0,0),1,1, color=color1, label='Rank 1'),
        plt.Rectangle((0,0),1,1, color=color2, label='Rank 2'),
        plt.Rectangle((0,0),1,1, color=color3, label='Rank 3')
    ], loc='best')

    bars =plt.bar(x, values, color=colors, width=0.8)
    unique_labels = ['Matrix 1', 'Matrix 2', 'Matrix 3', 'Matrix 4', 'Node']
    xticks_positions = [x[i * 3 + 1] for i in range(len(unique_labels))]
    plt.xticks(ticks=xticks_positions, labels=unique_labels)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.0f}%', ha='center', va='bottom')

    plt.ylabel('Proportion of body part in Matrix and Node appearing in Instruction(%)')
    plt.tight_layout()
    plt.savefig(output_path)


# Use command
# python utils/analyze.py /home/weihsin/projects/Evaluation/config_file/finetuneAttention_difference.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read configuration file and run setup.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    target_dir = read_config(args.config_path)
    target_path = target_dir + '/gpt_output.json'
    output_dir  = target_dir + '/analysis'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    matrix_calculate0, matrix_calculate1, matrix_calculate2, matrix_calculate3, node_calculate, data_number = analyze(target_path)
    draw_graph(matrix_calculate0, matrix_calculate1, matrix_calculate2, matrix_calculate3, node_calculate, output_dir, data_number)
    '''
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(os.path.dirname(file_path))
                file_name = file_name.split('e_')[1]
                joints = read_file(file_path)
                draw(joints, output_dir, file_name, video_dir,attention_node ,attention_matrix)
                Top3Node(file_name, attention_node, output_dir)
                Top3Link(file_name, attention_matrix, output_dir)
    '''