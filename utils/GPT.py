import json
import pickle
import openai
import os
import argparse
import yaml
import pickle
import json
from dotenv import load_dotenv

def call_gpt(prompt, api_key):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": f'Original instruction: "{prompt}"'}]
    )
    return response

def main(json_path,api_key):
    with open(json_path) as f:
        datas = json.load(f)
    results = []
 
    for data in datas:
        result = data.copy()
        print(f"Processing {result['video_name']}")
        prompt = f"""
                There are four matrix, every attention has two link. 
                Matrix 1 :
                {data['matrix']['0']['rank_1'][0]} and {data['matrix']['0']['rank_1'][1]}
                Matrix 2 :
                {data['matrix']['0']['rank_2'][0]} and {data['matrix']['0']['rank_2'][1]}
                Matrix 3 :
                {data['matrix']['0']['rank_3'][0]} and {data['matrix']['0']['rank_3'][1]}
                Matrix 4 :
                {data['matrix']['1']['rank_1'][0]} and {data['matrix']['1']['rank_1'][1]}
                Matrix 5 :
                {data['matrix']['1']['rank_2'][0]} and {data['matrix']['1']['rank_2'][1]}
                Matrix 6 :
                {data['matrix']['1']['rank_3'][0]} and {data['matrix']['1']['rank_3'][1]}
                Matrix 7 :
                {data['matrix']['2']['rank_1'][0]} and {data['matrix']['2']['rank_1'][1]}
                Matrix 8 :
                {data['matrix']['2']['rank_2'][0]} and {data['matrix']['2']['rank_2'][1]}
                Matrix 9 :
                {data['matrix']['2']['rank_3'][0]} and {data['matrix']['2']['rank_3'][1]}
                Matrix 10 :
                {data['matrix']['3']['rank_1'][0]} and {data['matrix']['3']['rank_1'][1]}
                Matrix 11 :
                {data['matrix']['3']['rank_2'][0]} and {data['matrix']['3']['rank_2'][1]}
                Matrix 12 :
                {data['matrix']['3']['rank_3'][0]} and {data['matrix']['3']['rank_3'][1]}
                Node 1 :
                {data['node']['rank_1']}
                Node 2 :
                {data['node']['rank_2']}
                Node 3 :
                {data['node']['rank_3']}
                If at least one body part mentioned in the matrix or node appears in the instruction, 
                or is related to a body part mentioned in the given instruction:
                {data['instruction']}.
                list its corresponding matrix number or node number and also list the names of the matched body parts.
                Reply like the following:
                "Matrix 2, Matrix 10, Node 2, Node 3, right ankle, left collar, left wrist"
                If there are no matches, then reply "None".
                """
 
        response = call_gpt(prompt, api_key)
        result['response'] = response.choices[0].message.content
        results.append(result)                                                                                              
        
    return results

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = config['output_dir']
    return output_dir 

# Use command
# python utils/GPT.py /home/weihsin/projects/Evaluation/config_file/finetuneAttention.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read configuration file and run setup.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    output_dir = read_config(args.config_path)
    json_path = output_dir + '/gpt_input.json'
    output_path = output_dir + '/gpt_output.json'
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the .env file
    api_key = os.getenv('OPENAI_KEY')

    results = main(json_path,api_key)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    