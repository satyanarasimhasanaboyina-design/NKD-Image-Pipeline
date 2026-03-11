

import os
import pandas as pd
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

# === CONFIG ===
csv_path = "E:/CaseStudy/engineered_prompts_advanced.xlsx"  # or use the German prompt version
prompt_column = "Prompt_Final"
negative_prompt_column = "Negative_Prompt"
generated_image_dir = "generated_images_lora_sdxl_adv_NLP"
generated_prompt_dir = "generated_prompts_lora_sdxl_adv_NLP"
num_images = 4
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_weights_path = "E:/CaseStudy/lora_outputs/sdxl_lora/checkpoint-500"
guidance_scale = 8

# === Create Output Folders ===
os.makedirs(generated_image_dir, exist_ok=True)
os.makedirs(generated_prompt_dir, exist_ok=True)

# === Load Prompt CSV ===
df = pd.read_excel(csv_path)

# === Load SDXL Pipeline and Apply LoRA ===
print(" Loading SDXL base model and applying LoRA...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Load LoRA weights directory (not the individual file)
pipe.load_lora_weights(lora_weights_path)
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

    print(f" Generated SDXL image for: {name} (row {i})")

print("\n All images generated using SDXL + LoRA.")
