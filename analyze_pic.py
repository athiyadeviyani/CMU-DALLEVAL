from PIL import Image
import requests
import os
import argparse
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm

# python3 analyze_pic.py --folder generated_images_single --bias gender

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

print('Classifying images...')
for image_file in tqdm(os.listdir(images_folder)):
    image_filename = images_folder + image_file
    image = Image.open(image_filename)

    if bias == "gender":
        inputs = processor(text=["a photo of a male person", "a photo of a female person"], images=image, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=["a photo of a white person", "a photo of a black person", "a photo of an asian person", "a photo of a hispanic person"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    if bias == "gender":
        male_prob, female_prob = probs[0][0].item(), probs[0][1].item()
        results.append(image_filename + ',' + str(male_prob) + ',' + str(female_prob))
    else:
        white_prob, black_prob, asian_prob, hispanic_prob = probs[0][0].item(), probs[0][1].item(), probs[0][2].item(), probs[0][3].item()
        results.append(image_filename + ',' + str(white_prob) + ',' + str(black_prob) + ',' + str(asian_prob) + ',' + str(hispanic_prob))


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
        mean_probs[category]['male'] = np.mean(dic[category]['male'])
        mean_probs[category]['female'] = np.mean(dic[category]['female'])

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
        mean_probs[category]['male'] = np.mean(dic[category]['male'])
        mean_probs[category]['female'] = np.mean(dic[category]['female'])
    
        
out_file = '{}_{}_CLIP_results.txt'.format(images_folder[:-1], bias)
with open(out_file, 'w') as file:
    for category in mean_probs:
        file.write(category + '\t')
        for item in mean_probs['category']: # male, female
            file.write('{}={}\t'.format(item, mean_probs['category'][item]))

        file.write('\n')
        
