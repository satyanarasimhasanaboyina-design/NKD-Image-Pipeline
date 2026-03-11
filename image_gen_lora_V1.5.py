# -*- coding: utf-8 -*-
"""
Generate images using LoRA fine-tuned weights (Stable Diffusion v1.5)
Author: SHAIK RIFSHU
"""

import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# === CONFIG ===
csv_path = "E:/CaseStudy/engineered_prompts_adv_all.xlsx"
prompt_column = "Prompt_Final"
negative_prompt_column = "Negative_Prompt"
generated_image_dir = "generated_images_lora_NLP_Adv"
generated_prompt_dir = "generated_prompts_lora_NLP_Adv"
num_images = 4
base_model = "runwayml/stable-diffusion-v1-5"
lora_weights_path = "E:/CaseStudy/outputs/lora_sd15"
guidance_scale = 8.5

# === Create Output Folders ===
os.makedirs(generated_image_dir, exist_ok=True)
os.makedirs(generated_prompt_dir, exist_ok=True)

# === Load Prompt CSV ===
df = pd.read_excel(csv_path)

# === Load Base Pipeline and Apply LoRA ===
print(" Loading Stable Diffusion v1.5 and applying LoRA weights...")
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# Load LoRA weights (directory, not the .safetensors file directly)
pipe.unet.load_attn_procs(lora_weights_path)
print(f" LoRA weights loaded from: {lora_weights_path}")

# === Generate Images ===
for i, row in df.iterrows():
    name = row.get('Name', f"item_{i}")[:40].replace("/", "_").replace("\\", "_")
    prompt = row.get(prompt_column, "").strip()
    negative_prompt = row.get(negative_prompt_column, "").strip()

    if not prompt:
        print(f" Skipping row {i}: Empty prompt")
        continue

    # Save prompt text
    with open(f"{generated_prompt_dir}/{i:03d}_{name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\nNegative Prompt: {negative_prompt}")

    # Generate image(s)
    for j in range(num_images):
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale).images[0]
        image.save(f"{generated_image_dir}/{i:03d}_{name}_{j+1}.png")

    print(f" Generated image for: {name} (row {i})")

print("\n All images generated using LoRA fine-tuned weights.")
