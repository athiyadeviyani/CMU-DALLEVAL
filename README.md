# CMU-DALLEVAL

Quick run
```
bash install.sh
bash setup.sh
python3 generate.py --samples {no. of prompts} --prompt_length {length of prompt} --num_images {no. of images/prompt} --model stablediffusion --outdir {output directory}
python3 analyze.py --bias {gender/race/skin} --folder {source folder containing images}
```
