# CMU-DALLEVAL

Set up environment
```
bash install.sh
bash setup.sh
```

Generate images (images will be saved to generated_images, image-prompt index will be saved to prompts.txt in the same folder)
```
tmux
python3 generate.py --num_prompts {number of prompts, max=145} \
                    --num_styles {number of artistic styles to sammple, max=50} \
                    --num_images {number of images per full prompt} \
                    --outdir {output directory, default=generated_images} 
```

Analyze with CLIP (will print a table and save a CSV file)
```
python3 analyze.py --bias {gender/race/skin} \
                   --folder {source folder containing images, default=generated_images}
```

Example
```
python3 generate.py --num_prompts 10 --num_styles 10 --num_images 5 --outdir generated_images
python3 analyze.py --bias gender --folder generated_images
```
