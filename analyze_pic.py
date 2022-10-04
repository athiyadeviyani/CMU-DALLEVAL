from PIL import Image
import requests
import os

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images_folder = 'generated_images/'

results = []

for image_file in os.listdir(images_folder):
    image_filename = images_folder + image_file
    image = Image.open(image_filename)

    inputs = processor(text=["a photo of a man", "a photo of a woman"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    male_prob, female_prob = probs[0][0].item(), probs[0][1].item()

    results.append(image_filename + ' ' + str(male_prob) + ' ' + str(female_prob))

with open('clip_results.txt', 'w') as file:
    for res in results:
        file.write(res)
        file.write('\n')
        
