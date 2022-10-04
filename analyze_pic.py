from PIL import Image
import requests
import os
import argparse
from transformers import CLIPProcessor, CLIPModel

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

for image_file in os.listdir(images_folder):
    image_filename = images_folder + image_file
    image = Image.open(image_filename)

    if bias == "gender":
        inputs = processor(text=["a photo of a male person", "a photo of a female person"], images=image, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=["a photo of a white person", "a photo of a black person", "a photo of an asian person", "a photo of a hispanic person"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    male_prob, female_prob = probs[0][0].item(), probs[0][1].item()

    results.append(image_filename + ' ' + str(male_prob) + ' ' + str(female_prob))

with open(images_folder + '_clip_results.txt', 'w') as file:
    for res in results:
        file.write(res)
        file.write('\n')
        
