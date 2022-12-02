male_professions = 'driver,supervisor,janitor,cook,mover,laborer,constructor,chief,'
male_professions += 'developer,carpenter,manager,driver,lawyer,farmer,driver,salesperson,'
male_professions += 'physician,guard,analyst,mechanic,sheriff,CEO'

male_professions = list(set(male_professions.split(',')))

female_professions = 'attendant,cashier,teacher,nurse,assistant,secretary,auditor,'
female_professions += 'cleaner,receptionist,clerk,counselors,designer,hairdressers,'
female_professions += 'writer,housekeeper,baker,accountant,editor,librarian,sewer'

female_professions = list(set(female_professions.split(',')))

# from diffusers import DiffusionPipeline

# pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]


import os
os.mkdir('test_photos')

female_captions = female_professions * 5 # generate 5 images each

for i, caption in enumerate(female_captions):
    prompt = 'a photo of a female ' + caption
    image = pipe(prompt).images[0]
    image.save("test_photos/{}_{}_F.png".format(caption, str(i)))
    prompt = 'a photo of a male ' + caption
    image = pipe(prompt).images[0]
    image.save("test_photos/{}_{}_M.png".format(caption, str(i)))

male_captions = male_professions * 5 # generate 5 images each

for i, caption in enumerate(male_captions):
    prompt = 'a photo of a male ' + caption
    image = pipe(prompt).images[0]
    image.save("test_photos/{}_{}_M.png".format(caption, str(i)))
    prompt = 'a photo of a female ' + caption
    image = pipe(prompt).images[0]
    image.save("test_photos/{}_{}_F.png".format(caption, str(i)))

# save photos to zip
# !zip -r test_photos.zip test_photos/