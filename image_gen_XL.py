# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:40:05 2025

@author: SHAIK RIFSHU
"""

import pandas as pd
import os
import time
from diffusers import StableDiffusionXLPipeline
import torch

# Load the engineered prompts
csv_path = "E:/CaseStudy/engineered_prompts_adv_all.xlsx"
df = pd.read_excel(csv_path)
 
# Create output directories
os.makedirs("generated_prompts_adv_NLP_xl", exist_ok=True)
os.makedirs("generated_images_adv_NLP_xl", exist_ok=True)

# Configuration
prompt_column = "Prompt_Final"
negative_prompt_column = "Negative_Prompt"
num_images = 4  # number of images per prompt

# Load SDXL pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# Iterate through prompts
for i, row in df.iterrows():
    product_name = row["Name"][:40].replace("/", "_").replace("\\", "_")
    prompt = row[prompt_column]
    negative_prompt = row[negative_prompt_column]

    if not isinstance(prompt, str) or not prompt.strip():
        print(f" Skipping empty prompt at row {i}")
        continue

    # Save prompt to a text file
    prompt_filename = f"generated_prompts_adv_NLP_xl/{i:03d}_{product_name}.txt"
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n\nNegative Prompt: {negative_prompt}")

    # Generate and save images
    for j in range(num_images):
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=8.5).images[0]
        image.save(f"generated_images_adv_NLP_xl/{i:03d}_{product_name}_{j+1}.png")

    print(f" Generated image for {product_name} (row {i})")
    time.sleep(0.2)

print("\n All prompts processed and images generated.")
