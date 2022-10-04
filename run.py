import transformers
from diffusers import DiffusionPipeline
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as img
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=str, default="10")
parser.add_argument("--length", type=str, default="50")
parser.add_argument("--prompt", type=str, default="simple")  # magic
parser.add_argument("--white", action='store_true')
args = parser.parse_args()
samples = int(args.samples)
prompt_gen = args.prompt
white_bg = args.white
prompt_length = int(args.length)


#############################################################################
############################# PROMPT GENERATION #############################
#############################################################################

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


if prompt_gen == "magic":

    tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
    model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")

    prompt_generator = transformers.pipeline('text-generation', 
                                    model='Gustavosta/MagicPrompt-Stable-Diffusion', 
                                    tokenizer=tokenizer)

    modified_prompts = []

    print('Generating prompts...')
    for prompt in tqdm(raw_prompts):
        generated_prompt = prompt_generator(prompt, pad_token_id=tokenizer.eos_token_id, max_length=prompt_length)
        generated_prompt = generated_prompt[0]['generated_text'].replace('\n', '')
        modified_prompts.append(generated_prompt)

else: # simple
    art_styles = []
    with open('art_styles.txt', 'r') as f:   
        for line in f.readlines():
            art_styles.append(line.replace('\n',''))

    
    modified_prompts = []

    print('Generating prompts...')
    for prompt in tqdm(raw_prompts):
        style = np.random.choice(art_styles, 1)[0]
        new_prompt = prompt + style
        if white_bg:
            new_prompt = new_prompt + ', white background'
        modified_prompts.append(new_prompt)

#############################################################################
############################## STABLE DIFFUSION #############################
#############################################################################

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline = pipeline.to("cuda")

# First-time "warmup" pass (see explanation above)
# prompt = "a photo of an astronaut riding a horse on mars"
# _ = pipeline(prompt, num_inference_steps=1)
# # Results match those from the CPU device after the warmup pass.
# image = pipeline(prompt).images[0]

sample_prompts = np.random.choice(modified_prompts, samples)

generated_grids = []
num_images = 5  # number of images per prompt

print('Generating images...')
for prompt in tqdm(sample_prompts):
    plt.figure(figsize=(30, 10))
    if prompt_gen == 'simple':
        filename = 'generated_images_simple/'+'_'.join(prompt.split(',')[0].split(' ')) + '.png'
    elif white_bg:
        filename = 'generated_images_white/'+'_'.join(prompt.split(',')[0].split(' ')) + '.png'
    else:
        filename = 'generated_images/'+'_'.join(prompt.split(',')[0].split(' ')) + '.png'

    for i in range(num_images):
        image = pipeline(prompt).images
        plt.title(prompt + str(i))
        image = img.imread(image)
        plt.imshow(image)
        filename = 'generated_images/'+'_'.join(prompt.split(',')[0].split(' ')) + str(i) + '.png'
        plt.savefig(filename)
        plt.close()