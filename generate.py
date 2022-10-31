import os
import transformers
import argparse
import numpy as np

from tqdm import tqdm
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# sample
# python3 generate.py --samples 20 --prompt_length 50 --num_images 5 --model stablediffusion --outdir gen_imgs_sd
# python3 generate.py --samples 50 --num_images 10 --outdir gen_imgs_disney --disney

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=10)  # number of prompts
parser.add_argument("--prompt_length", type=int, default=50)
parser.add_argument("--num_images", type=int, default=5)  # number of images per prompt
parser.add_argument("--model", type=str, default='stablediffusion')  # stablediffusion/dalle
parser.add_argument("--outdir", type=str, default='generated_images')  # output folder
parser.add_argument("--disney", action='store_true')

args = parser.parse_args()
samples = args.samples
prompt_length = args.prompt_length
num_images = args.num_images
model = args.model
outdir = args.outdir
disney = args.disney

if outdir not in os.listdir():
    os.mkdir(outdir)

pretrained_model = "Gustavosta/MagicPrompt-Stable-Diffusion" if model == "stablediffusion" else "Gustavosta/MagicPrompt-Dalle"

# Select sample prompts randomly from template

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

if disney:
    chosen_prompts = np.random.choice(raw_prompts, samples)
    sample_prompts = []

    print('Generating prompts...')
    for prompt in tqdm(list(chosen_prompts)*num_images):
        new_prompt = prompt + 'modern disney style'
        sample_prompts.append(new_prompt)

else:
    print('Loading {}...'.format(pretrained_model))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model)

    prompt_generator = transformers.pipeline('text-generation', 
                                    model=pretrained_model, 
                                    tokenizer=tokenizer)

    # randomly pick samples and generate
    chosen_prompts = np.random.choice(raw_prompts, samples)
    sample_prompts = []

    print('Generating prompts...')
    for prompt in tqdm(list(chosen_prompts)*num_images):
        generated_prompt = prompt_generator(prompt, pad_token_id=tokenizer.eos_token_id, min_length=prompt_length//2, max_length=prompt_length)
        generated_prompt = generated_prompt[0]['generated_text'].replace('\n', '')
        sample_prompts.append(generated_prompt)


## Generate images
if disney:  # fix the style of the art
    print('Loading {}...'.format("nitrosocke/mo-di-diffusion"))
    pipeline = DiffusionPipeline.from_pretrained("nitrosocke/mo-di-diffusion")
    pipeline = pipeline.to("cuda")
else:
    print('Loading {}...'.format("CompVis/stable-diffusion-v1-4"))
    pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipeline = pipeline.to("cuda")

print('Generating images...')
for i, prompt in tqdm(enumerate(sample_prompts)):
    image = pipeline(prompt).images[0]
    image.save("{}/{}.png".format(outdir, i))

with open('{}/prompts.txt'.format(outdir), 'w') as f:
    for i, prompt in tqdm(enumerate(sample_prompts)):
        f.write('{}\t{}\n'.format(i, prompt))