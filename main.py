# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 22:05:40 2025

@author: SHAIK RIFSHU
"""


import os
import pandas as pd
from prompt_generation import prompt_template, prompt_advanced
from image_generation import sd15, sd15_lora, sdxl, sdxl_lora

# === Step 1: Ask user for input ===
print(" Please enter the folder path where your .xlsx files are stored:")
folder = input("Folder path: ").strip()

print("\n Choose prompt method:")
print("1. Template method (multiple prompt styles)")
print("2. Prompt Advance (short SDXL-style prompt)")
prompt_choice = input("Enter 1 or 2: ").strip()

print("\n Choose image model:")
print("1. SD1.5")
print("2. SD1.5 + LoRA")
print("3. SDXL")
print("4. SDXL + LoRA")
model_choice = input("Enter 1, 2, 3 or 4: ").strip()

num_images = int(input("\n How many images per prompt? (e.g. 2): ").strip())
guidance_scale = float(input(" Guidance scale (e.g. 7.5): ").strip())

# === Step 2: Generate prompts ===
if prompt_choice == "1":
    prompts_df = prompt_template.generate_prompts(folder)
    prompt_column = "Prompt_Manual"
elif prompt_choice == "2":
    prompts_df = prompt_advanced.generate_prompts(folder)
    prompt_column = "Prompt_Final"
else:
    print(" Invalid prompt method selected.")
    exit()

# Save prompts
os.makedirs("Final_outputs/prompts", exist_ok=True)
prompt_file = f"Final_outputs/prompts/prompts_method_{prompt_choice}.xlsx"
prompts_df.to_excel(prompt_file, index=False)
print(f"\n Prompts saved to {prompt_file}")

# === Step 3: Generate images ===
if model_choice == "1":
    model = sd15.ImageGenerator(prompts_df, prompt_column, num_images, guidance_scale)
elif model_choice == "2":
    model = sd15_lora.ImageGenerator(prompts_df, prompt_column, num_images, guidance_scale)
elif model_choice == "3":
    model = sdxl.ImageGenerator(prompts_df, prompt_column, num_images, guidance_scale)
elif model_choice == "4":
    model = sdxl_lora.ImageGenerator(prompts_df, prompt_column, num_images, guidance_scale)
else:
    print(" Invalid model selected.")
    exit()

model.generate_images()
print(" Image generation completed!")
