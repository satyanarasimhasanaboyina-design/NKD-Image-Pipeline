HOW TO RUN IMAGE GENERATION PIPELINE
------------------------------------

1. Install Requirements:
   pip install pandas torch diffusers transformers deep-translator yake openpyxl

2. Run Prompt Engineering:

   # Simple Prompts (Manual, Simple, YAKE)
   python prompt_1.py

   # Advanced Prompts (Sentiment, Summary, YAKE)
   python prompt_advanced1.py

3. Locate Generated Excel File:

   # If you ran prompt_1.py:
   --> engineered_prompts_all.xlsx

   # If you ran prompt_advanced1.py:
   --> engineered_prompts_advanced_3.xlsx

   Copy the full path of the generated Excel file.
   Example: E:/CaseStudy/engineered_prompts_all.xlsx

4. Update Image Generation Script:

   Open any of these files:
     - image_gen_v1.5.py
     - image_gen_lora_V1.5.py
     - image_gen_XL.py
     - image_gen_lora_sdxl.py

   Replace the line starting with:
     csv_path = ...
   with your copied path

   Also set the correct column name:
     prompt_column = "Prompt_Manual"   # Or Prompt_Simple, Prompt_NLP, Prompt_Final
     negative_prompt_column = "Negative_Prompt"

5. Run Image Generation:

   # For Stable Diffusion v1.5
   python image_gen_v1.5.py

   # For SD v1.5 + LoRA
   python image_gen_lora_V1.5.py

   # For Stable Diffusion XL (SDXL)
   python image_gen_XL.py

   # For SDXL + LoRA
   python image_gen_lora_sdxl.py