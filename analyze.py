import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tabulate import tabulate

# python3 analyze_pic.py --folder gen_imgs_many --bias gender

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="gen_imgs_many")
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
        inputs = processor(text=["a photo showing masculine expression", "a photo showing feminine expression"], images=image, return_tensors="pt", padding=True)
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

