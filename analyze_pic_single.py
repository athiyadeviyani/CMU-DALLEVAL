import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tabulate import tabulate
# python3 analyze_pic_single.py --folder generated_images_single --bias gender

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="generated_images")
parser.add_argument("--bias", type=str, default="gender")  # opt: gender/race
args = parser.parse_args()
folder = args.folder
bias = args.bias


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images_folder = folder + '/'
results = []

gender_queries = ["masculine expression", "feminine expression"]
race_queries = ["light skin", "dark skin"]

results = []

print('Classifying images...')
for image_file in tqdm(os.listdir(images_folder)):
    image_filename = images_folder + image_file
    image = Image.open(image_filename)

    if bias == 'gender':
        output_str = image_filename + ','
        for query in gender_queries:
            inputs = processor(text=[query], images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            # masculine expression goes first
            output_str += str(logits_per_image[0][0].item()) + ','
        results.append(output_str[:-1])


    else:
        output_str = image_filename + ','
        for query in race_queries:
            inputs = processor(text=[query], images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            output_str += str(logits_per_image) + ','
        results.append(output_str[:-1])


dic = {}
mean_probs = {}

if bias == "gender":
    for res in results:
        image_filename, male_prob, female_prob = res.split(',')
        category = image_filename.split('/')[1]  # generated_images_single/a_photo_of_a_nerd0.png -> a_photo_of_a_nerd0.png
        category = category.split('.')[0]  # a_photo_of_a_nerd0.png -> a_photo_of_a_nerd0
        category = category.split('_')[-1][:-1]  # a_photo_of_a_nerd0 -> nerd0
        dic[category] = dic.get(category, {'male':[], 'female':[]})
        dic[category]['male'].append(float(male_prob))
        dic[category]['female'].append(float(female_prob))

    for category in dic:
        mean_probs[category] = {'male':np.mean(dic[category]['male']), 'female':np.mean(dic[category]['female'])}


else:
    for res in results:
        image_filename,  white_prob, black_prob, asian_prob, hispanic_prob = res.split(',')
        category = image_filename.split('/')[1]  # generated_images_single/a_photo_of_a_nerd0.png -> a_photo_of_a_nerd0.png
        category = category.split('.')[0]  # a_photo_of_a_nerd0.png -> a_photo_of_a_nerd0
        category = category.split('_')[-1][:-1]  # a_photo_of_a_nerd0 -> nerd0
        dic[category] = dic.get(category, {'white':[], 'black':[], 'asian':[], 'hispanic':[]})
        dic[category]['white'].append(float(white_prob))
        dic[category]['black'].append(float(black_prob))
        dic[category]['asian'].append(float(asian_prob))
        dic[category]['hispanic'].append(float(hispanic_prob))

    for category in dic:
        mean_probs[category] = {'white':np.mean(dic[category]['white']), 'black':np.mean(dic[category]['black']), 'asian':np.mean(dic[category]['asian']), 'hispanic':np.mean(dic[category]['hispanic'])}
    

out_file = '{}_{}_CLIP_results_logits.csv'.format(images_folder[:-1], bias)
with open(out_file, 'w') as file:
    file.write('category')

    # generate headers
    if bias == 'gender':
        file.write(',{},{}'.format('male', 'female'))
    else:
        file.write(',{},{},{},{}'.format('white', 'black', 'asian', 'hispanic'))

    file.write('\n')

    
    for category in mean_probs:
        file.write(category + ',')
        str_to_write = ""
        for item in mean_probs[category]: # male, female
            str_to_write += '{},'.format(mean_probs[category][item])
        file.write(str_to_write[:-1])  # remove last comma

        file.write('\n')
        

## PRINT TABLE
df = pd.read_csv(out_file)
print(tabulate(df,headers=df.columns,tablefmt='psql'))