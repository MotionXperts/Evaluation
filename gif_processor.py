from PIL import Image, ImageDraw, ImageFont, ImageSequence
import os, json, cv2, textwrap
import numpy as np
def add_subtitle_to_gif(input_path, output_path, subtitle_text, result_text, max_chars_per_line=40):
    gif = Image.open(folder_path + '/' +input_path)
    #print(subtitle_text)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.3
    color = (0,0,0)
    thickness = 2
    lineType = cv2.LINE_AA

    frame_count = 0
    
    img_list = []
    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert('RGBA')
        opencv_img = np.array(frame, dtype=np.uint8)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGBA2BGRA)

        if frame_count < gif.n_frames/2:
            text = subtitle_text[0]
        else :
            text = result_text
        '''
        elif frame_count < gif.n_frames/5*2:
            text = subtitle_text[1]
        elif frame_count < gif.n_frames/5*3:
            text = subtitle_text[2]
        elif frame_count < gif.n_frames/5*4:
            text = subtitle_text[3]
        else :
            text = result_text
        '''
        frame_count += 1

        lines = textwrap.wrap(text, width=max_chars_per_line)
        text_height = 0
        for line in lines:
            (width, baseline), _ = cv2.getTextSize(line, fontFace, fontScale, thickness)
            text_height += baseline

        x, y= 50, 100

        for line in lines:
            (width, baseline), _ = cv2.getTextSize(line, fontFace, fontScale, thickness)
            baseline += 15
            cv2.putText(opencv_img, line, (x, y), fontFace, fontScale, color, thickness, lineType)
            y += baseline

        img_list.append(opencv_img)

    output = []
    for i in img_list:
        img = i
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(img) 
        img = img.convert('RGB')
        output.append(img)            

    subword = "attention"
    if subword in input_path :
        output[0].save(output_path + input_path.split("_")[0] + "_" + input_path.split("_")[1] + "attention_annotated.gif", save_all=True, append_images=output[1:], loop=0, disposal=0, fps=100)
    else :
        output[0].save(output_path + input_path.split("_")[0] + "_" + input_path.split("_")[1] + "_annotated.gif", save_all=True, append_images=output[1:], loop=0, disposal=0, fps=100)
    cv2.destroyAllWindows()
    #return output

folder_path = "./projects/MotionExpert/STAGCN_output_finetune/finetuenattention"
file_list = [f for f in os.listdir(folder_path) if f.endswith('.gif')]

output_gif_path = "./projects/MotionExpert/STAGCN_output_finetune/annotation/"

subtitles = []
json_files = ["results_epoch3.json"]
messages = ["Predict : "]

count = len(file_list)

for f in file_list:
    for i in range(1):
        json_file_path = './projects/MotionExpert/STAGCN_output_finetune/' + json_files[i]
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
        subtitles.append(messages[i] + json_data[f.split("_")[0]+"_"+f.split("_")[1]])

    json_file_path = './projects/MotionExpert/STAGCN_output_finetune/ground_truth.json'
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
    subtitle_result = "Ground truth : " + json_data[f.split("_")[0]+"_"+f.split("_")[1]]
    print("file_anme",f)
    add_subtitle_to_gif(f, output_gif_path, subtitles, subtitle_result)
    subtitles.clear()

    count -= 1
    print(f.split("_")[0]+"_"+f.split("_")[1] + " is done, " + str(count) + " left to be processed.")

print("done")
