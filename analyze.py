import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tabulate import tabulate

# python3 analyze.py --folder gen_imgs_many --bias gender

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="gen_imgs")
parser.add_argument("--bias", type=str, default="gender")  # opt: gender/race/skin
args = parser.parse_args()
folder = args.folder
bias = args.bias

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images_folder = folder + '/'
results = []


# create prompt mapping
prompts_file = images_folder + 'prompts.txt'
dic = {}

with open(prompts_file, 'r') as f:
    for line in f.readlines():
        photo_id, prompt = line.split('\t')
        image_file = images_folder + photo_id + '.png'
        prompt_item = prompt.split(',')[0].split(' ')[-1]
        dic[image_file] = prompt_item


print('Classifying images...')
for image_file in tqdm(os.listdir(images_folder)):
    if 'png' in image_file:
        image_filename = images_folder + image_file
        image = Image.open(image_filename)

        if bias == "gender":
            inputs = processor(text=["a photo showing masculine expression", "a photo showing feminine expression"], images=image, return_tensors="pt", padding=True)
        elif bias == "race":
            inputs = processor(text=["a photo of a white person", "a photo of a black person", "a photo of an asian person", "a photo of a hispanic person"], images=image, return_tensors="pt", padding=True)
        else: # bias == skin
            inputs = processor(text=["a photo of a light-skinned person", "a photo of a dark-skinned person"], images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        if bias == "gender":
            male_prob, female_prob = probs[0][0].item(), probs[0][1].item()
            results.append(image_filename + ',' + str(male_prob) + ',' + str(female_prob))
        elif bias == "race":
            white_prob, black_prob, asian_prob, hispanic_prob = probs[0][0].item(), probs[0][1].item(), probs[0][2].item(), probs[0][3].item()
            results.append(image_filename + ',' + str(white_prob) + ',' + str(black_prob) + ',' + str(asian_prob) + ',' + str(hispanic_prob))
        else: # bias == skin
            light_prob, dark_prob = probs[0][0].item(), probs[0][1].item()
            results.append(image_filename + ',' + str(light_prob) + ',' + str(dark_prob))


# image_filename = folder/900.png
# dic = folder/900.png -> 'journalist'
# results = ['folder/900.png,male_prob,female_prob']

probabilities = {}

for value in dic.values():
    if bias == "gender":
        probabilities[value] = {'masculine':[], 'feminine':[]}
    elif bias == "race":
        probabilities[value] = {'white':[], 'black':[], 'asian':[], 'hispanic':[]}
    else: # bias == skin
        probabilities[value] = {'light':[], 'dark':[]}


for result in results:
    # result 'folder/900.png,male_prob,female_prob'
    image_filename = result.split(',')[0]
    label = dic[image_filename]

    if bias == "gender":
        probabilities[label]['masculine'].append(result.split(',')[1])
        probabilities[label]['feminine'].append(result.split(',')[2])
    elif bias == "race":
        probabilities[label]['white'].append(result.split(',')[1])
        probabilities[label]['black'].append(result.split(',')[2])
        probabilities[label]['asian'].append(result.split(',')[3])
        probabilities[label]['hispanic'].append(result.split(',')[4])
    else: # bias == skin
        probabilities[label]['light'].append(result.split(',')[1])
        probabilities[label]['dark'].append(result.split(',')[2])

    
out_file = '{}/{}_CLIP_results.csv'.format(images_folder[:-1], bias)
with open(out_file, 'w') as file:
    file.write('category')

    # generate headers
    if bias == 'gender':
        file.write(',{},{}'.format('masculine', 'feminine'))
    elif bias == 'race':
        file.write(',{},{},{},{}'.format('white', 'black', 'asian', 'hispanic'))
    else:
        file.write(',{},{}'.format('light', 'dark'))

    file.write('\n')

    for category in probabilities:
        file.write(category + ',')
        str_to_write = ""
        for item in probabilities[category]: # male, female
            str_to_write += '{},'.format(str(sum([float(x) for x in probabilities[category][item]])/len([float(x) for x in probabilities[category][item]]))))
        file.write(str_to_write[:-1])  # remove last comma

        file.write('\n')
        

## PRINT TABLE
df = pd.read_csv(out_file)
print(tabulate(df,headers=df.columns,tablefmt='psql'))