# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 04:55:20 2025

@author: SHAIK RIFSHU
"""



import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration
from deep_translator import MyMemoryTranslator
import yake

# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(" GPU:", torch.cuda.get_device_name(0))

# === Load Dataset ===
df = pd.read_excel("E:/CaseStudy/NKD Image generation datasets-20250505/article_descriptions 1.xlsx")
df = df.dropna(subset=['name', 'description', 'model_name', 'model_descr']).copy()

# === NLP Models ===
# 1. Sentiment (Multilingual BERT)
tokenizer_sent = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model_sent = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(device)

def get_sentiment(text):
    inputs = tokenizer_sent(text, return_tensors="pt", truncation=True).to(device)
    outputs = model_sent(**inputs)
    score = torch.argmax(outputs.logits).item()
    return ["sehr negativ", "negativ", "neutral", "positiv", "sehr positiv"][score]

# 2. Summarizer (BART)
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

def summarize_and_backtranslate(text):
    try:
        # Translate to English
        english = MyMemoryTranslator(source='de', target='en').translate(text)
        # Summarize
        inputs = tokenizer_bart([english], max_length=1024, return_tensors='pt', truncation=True).to(device)
        ids = model_bart.generate(inputs['input_ids'], max_length=50, min_length=10, num_beams=4, early_stopping=True)
        summary_en = tokenizer_bart.decode(ids[0], skip_special_tokens=True)
        # Translate back to German
        summary_de = MyMemoryTranslator(source='en', target='de').translate(summary_en)
        return summary_de
    except:
        return "Zusammenfassung nicht verfügbar"

# 3. Keyword Extractor
kw_extractor = yake.KeywordExtractor(lan="de", n=1, top=5)

# === Prompt Construction ===
def build_prompt_de(article_type, name, keywords, summary):
    # Prompt in German only
    base = f"Studioaufnahme eines {article_type.lower()} namens '{name}', mit Merkmalen wie {keywords}. {summary}. Weißer Hintergrund, keine Person."
    words = base.split()
    if len(words) > 70:
        base = f"{article_type} Produktbild mit: {keywords}. Hintergrund weiß, ohne Mensch."
    return base

negative_prompt_de = "Mensch, Gesicht, Modell, Haut, Hände, unscharf, Wasserzeichen, Text, abgeschnitten, Logo"

# === Generate All Prompts ===
prompt_data = []

for _, row in df.iterrows():
    name = row['name']
    desc_de = row['description']
    article_type = row['model_name']
    model_descr = row['model_descr']

    try:
        summary = summarize_and_backtranslate(desc_de)
    except:
        summary = "Zusammenfassung nicht verfügbar"

    try:
        sentiment = get_sentiment(desc_de)
    except:
        sentiment = "Unbekannt"

    try:
        keywords = ", ".join([kw for kw, _ in kw_extractor.extract_keywords(desc_de)])
    except:
        keywords = ""

    final_prompt = build_prompt_de(article_type, name, keywords, summary)

    prompt_data.append({
        "Name": name,
        "Article_Type": article_type,
        "Model_Descr": model_descr,
        "Description_German": desc_de,
        "Summary": summary,
        "Keywords": keywords,
        "Sentiment": sentiment,
        "Prompt_German": final_prompt,
        "Negative_Prompt": negative_prompt_de
    })

# === Save to CSV ===
df_out = pd.DataFrame(prompt_data)
df_out.to_csv("engineered_prompts_german.csv", index=False)
print("✅ German prompts saved to engineered_prompts_german.csv")

# === Sample Output ===
print("\n=== FIRST 3 GERMAN PROMPTS ===")
for i in range(min(3, len(df_out))):
    row = df_out.iloc[i]
    print(f"\n🔸 {row['Name']}")
    print(f"📘 {row['Description_German']}")
    print(f"🧠 Zusammenfassung: {row['Summary']}")
    print(f"🔍 Schlüsselwörter: {row['Keywords']}")
    print(f"😊 Stimmung: {row['Sentiment']}")
    print(f"🧩 Prompt:\n{row['Prompt_German']}")
