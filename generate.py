import os
import transformers
import argparse
import numpy as np

from tqdm import tqdm
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_prompts", type=int, default=10)  # number of prompts
parser.add_argument("--num_styles", type=int, default=10)  # number of styles per prompts
parser.add_argument("--num_images", type=int, default=5)  # number of images per prompt
# if prompts = 10, styles = 10, num_images = 5, total generated = 10 * 10 * 5 = 500
# parser.add_argument("--model", type=str, default='stablediffusion')  # stablediffusion/dalle
parser.add_argument("--outdir", type=str, default='generated_images')  # output folder

# sample: generate 10 prompts ("a photo of a nurse"), 
#   with 10 styles ("a photo of a nurse, digital art style"), 
#   5 images each (to control for irregularities)
#   stablediffusion model
#   save to generated_images
# python3 generate.py --num_prompts 10 --num_styles 10 --num_images 5 --outdir generated_images


# python3 generate.py --num_prompts 20 --num_styles 25 --num_images 5 --outdir generated_images

args = parser.parse_args()
num_prompts = args.num_prompts
num_styles = args.num_styles
num_images = args.num_images
outdir = args.outdir 

if outdir not in os.listdir():
    os.mkdir(outdir)

print('ARGS: num_prompts={}, num_styles={}, num_images={}, outdir={}'.format(
    num_prompts, num_styles, num_images, outdir))

############# select sample prompts randomly from template #############
print('Generating prompts...')

professions = []
with open('templates/professions.txt', 'r') as f:
    for line in f.readlines():
        professions.append(line.replace('\n',''))

objects = []
with open('templates/objects.txt', 'r') as f:
    for line in f.readlines():
        objects.append(line.replace('\n',''))

political = []
with open('templates/political.txt', 'r') as f:
    for line in f.readlines():
        political.append(line.replace('\n',''))

others = []
with open('templates/others.txt', 'r') as f:
    for line in f.readlines():
        others.append(line.replace('\n',''))

template_1 = 'a photo of a {}, '
template_2 = 'a person with a {}, '

raw_prompts = []

for person in others+political+professions:
    raw_prompts.append(template_1.format(person))

for item in objects:
    raw_prompts.append(template_2.format(item))

# sample
chosen_prompts = np.random.choice(raw_prompts, num_prompts)

print('Total prompts={}'. format(len(raw_prompts)))
print('Sampled prompts={}'.format(len(chosen_prompts)))


############# append style to prompt #############

styles = []
with open('templates/styles.txt', 'r') as f:
    for line in f.readlines():
        styles.append(line.replace('\n',''))

# sample
chosen_styles = np.random.choice(styles, num_styles)

print('Total styles={}' .format(len(styles)))
print('Sampled styles={}'.format(len(chosen_styles)))


############# prompts to generate #############

# total length = len(chosen_prompts) * len(chosen_styles) * num_images

prompts = []

for prompt in chosen_prompts:
    for style in chosen_styles:
        final_prompt = prompt + style + 'style'  # a photo of a nurse, digital art style
        prompts.append(final_prompt)

prompts = prompts * num_images

print('Generated {} prompts!'.format(len(prompts)))

############# generate images#############
pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline = pipeline.to("cuda")

print('Generating images to {}...'.format(outdir))
for i, prompt in tqdm(enumerate(prompts)):
    image = pipeline(prompt).images[0]
    image.save("{}/{}.png".format(outdir, i))

# save prompts
print('Prompts saved to {}/prompts.txt'.format(outdir))
with open('{}/prompts.txt'.format(outdir), 'w') as f:
    for i, prompt in tqdm(enumerate(prompts)):
        f.write('{}\t{}\n'.format(i, prompt))