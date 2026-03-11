# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:58:36 2025

@author: SHAIK RIFSHU
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration
from deep_translator import GoogleTranslator
import yake

# Detect GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(" GPU:", torch.cuda.get_device_name(0))

# Load Excel
df = pd.read_excel("E:/CaseStudy/NKD Image generation datasets-20250505/article_descriptions 3.xlsx")
df = df.dropna(subset=['name', 'description', 'model_name', 'model_descr']).copy()

# Initialize NLP tools
translator = GoogleTranslator(source='de', target='en')
kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=5)

# BERT Sentiment Classifier
tokenizer_sentiment = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(device)

def get_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True).to(device)
    outputs = model_sentiment(**inputs)
    score = torch.argmax(outputs.logits).item()
    return ["very negative", "negative", "neutral", "positive", "very positive"][score]

# BART Summarizer
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

def summarize_text(text):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(device)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Clean and sanitize prompt text
def sanitize_brand_terms(text):
    replacements = {
        "Mickey Mouse": "cartoon mouse character",
        "Disney": "cartoon character",
        "Marvel": "comic-style graphic",
        "Star Wars": "sci-fi themed print"
    }
    for brand, generic in replacements.items():
        text = text.replace(brand, generic)
    return text

def clean_text(text):
    text = sanitize_brand_terms(text)
    try:
        if any(ord(c) > 127 for c in text):
            text = translator.translate(text)
    except:
        pass
    fluff = ["pleasant", "optimal", "offers", "ensures", "feeling", "comfort", "comfortable", "high-quality", "quality", "wearing"]
    for word in fluff:
        text = text.replace(word, "")
    seen = set()
    deduped = []
    for word in text.split():
        if word.lower() not in seen:
            deduped.append(word)
            seen.add(word.lower())
    return " ".join(deduped).strip().replace("  ", " ")

# Prompt Assembly with token limit control
def build_prompt(name, article_type, keywords, tone, summary):
    name_cleaned = clean_text(name)
    summary_cleaned = clean_text(summary)
    keywords_cleaned = clean_text(keywords)

    base_prompt = (
        f"Studio-quality flat lay photo of a {article_type.lower()} titled '{name_cleaned}', highlighting key features such as {keywords_cleaned}. "
f"{summary_cleaned} Set against a clean white background. Catalog-style presentation with no humans, no models, and no mannequins."
)

    # Truncate to approx 70 tokens to avoid CLIP 77 token limit
    words = base_prompt.split()
    if len(words) > 70:
        base_prompt = " ".join(words[:70])

    return base_prompt

# Shared negative prompt
negative_prompt = "human, person, face, model, pose, mannequin, skin, hands, blurry, watermark, text, cropped"

# Process all rows
prompt_data = []
for idx, row in df.iterrows():
    name = row['name']
    article_type = row['model_name']
    desc_de = row['description']
    model_descr = row['model_descr']

    try:
        desc_en = translator.translate(desc_de)
    except:
        desc_en = "TRANSLATION_FAILED"

    if desc_en == "TRANSLATION_FAILED":
        prompt_data.append({**row, "Prompt_Final": "", "Summary": "", "Keywords": "", "Sentiment": "", "Negative_Prompt": negative_prompt})
        continue

    keywords = ", ".join([kw for kw, _ in kw_extractor.extract_keywords(desc_en)])
    sentiment = get_sentiment(desc_en)
    summary = summarize_text(desc_en)
    final_prompt = build_prompt(name, article_type, keywords, sentiment, summary)

    prompt_data.append({
        "Name": name,
        "Article_Type": article_type,
        "Description_German": desc_de,
        "Description_English": desc_en,
        "Model_Descr": model_descr,
        "Keywords": keywords,
        "Sentiment": sentiment,
        "Summary": summary,
        "Prompt_Final": final_prompt,
        "Negative_Prompt": negative_prompt
    })

# Save to CSV
df_out = pd.DataFrame(prompt_data)
df_out.to_excel("engineered_prompts_advanced_3.xlsx", index=False)
print(" Saved all advanced prompts to engineered_prompts_advanced.csv")

# Print samples
print("\n=== SAMPLE OUTPUT (First 3 Entries) ===")
for i in range(min(3, len(df_out))):
    row = df_out.iloc[i]
    print(f"\n Name: {row['Name']}")
    print(f" English Description: {row['Description_English']}")
    print(f" Summary (BART): {row['Summary']}")
    print(f" Sentiment (BERT): {row['Sentiment']}")
    print(f" Keywords (YAKE): {row['Keywords']}")
    print(f" Final Prompt:\n{row['Prompt_Final']}")
