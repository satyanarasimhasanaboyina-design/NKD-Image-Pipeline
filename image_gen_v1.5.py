# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:15:20 2025

@author: SHAIK RIFSHU
"""


import pandas as pd
import os
import time
import torch
from diffusers import StableDiffusionPipeline

# Load prompts
csv_path = "E:/CaseStudy/engineered_prompts_adv_all.xlsx"
df = pd.read_excel(csv_path)

# Output directories
os.makedirs("generated_prompts_v1_5_adv_NLP", exist_ok=True)
os.makedirs("generated_images_v1_5_adv_NLP", exist_ok=True)

# Config
prompt_column = "Prompt_Final"
negative_prompt_column = "Negative_Prompt"
num_images = 4

# Load SD v1.5 pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.enable_attention_slicing()

# Generate
for i, row in df.iterrows():
    product_name = row["Name"][:40].replace("/", "_").replace("\\", "_")
    prompt = row[prompt_column]
    negative_prompt = row[negative_prompt_column]

    if not isinstance(prompt, str) or not prompt.strip():
        print(f" Skipping empty prompt at row {i}")
        continue

    # Save prompt
    prompt_file = f"generated_prompts_v1_5_adv_NLP/{i:03d}_{product_name}.txt"
    with open(prompt_file, "w") as f:
        f.write(f"Prompt: {prompt}\n\nNegative Prompt: {negative_prompt}")

    # Generate image(s)
    for j in range(num_images):
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale= 8.5).images[0]
        image.save(f"generated_images_v1_5_adv_NLP/{i:03d}_{product_name}_{j+1}.png")

    print(f"Image generated for: {product_name} (row {i})")
    time.sleep(0.2)

print("\nAll prompts processed with SD v1.5.")
