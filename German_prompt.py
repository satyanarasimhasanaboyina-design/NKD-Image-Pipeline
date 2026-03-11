# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 04:41:37 2025

@author: SHAIK RIFSHU
"""

import pandas as pd
import yake
import re

# === Load Excel File ===
excel_path = "E:/CaseStudy/NKD Image generation datasets-20250505/article_descriptions 1.xlsx"
df = pd.read_excel(excel_path)
df = df.dropna(subset=['name', 'description', 'model_name', 'model_descr']).copy()

# === YAKE Keyword Extractor (German) ===
kw_extractor = yake.KeywordExtractor(lan="de", n=1, top=5)

# === Gemeinsamer Negativ-Prompt ===
negative_prompt_text = (
    "Mensch, Person, Gesicht, Kopf, Körper, Haut, Arme, Beine, Hände, Modell, Pose, Schaufensterpuppe, "
    "unscharf, Wasserzeichen, Text, abgeschnitten, Logo"
)

# === Helper to clean repeated/bullet description ===
def clean_description(text):
    if not isinstance(text, str):
        return ""
    
    # Split into lines and filter out bullets/lists
    lines = text.strip().split("\n")
    paragraph = []
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith(("-", "–", "•", "\t")) or line_strip == "":
            break
        paragraph.append(line_strip)

    cleaned = " ".join(paragraph)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip().rstrip(".") + "."

# === Prompt Templates ===
def manual_template_prompt(article_type, desc_de):
    desc_clean = clean_description(desc_de)
    return (
        f"Studiofoto eines einzelnen {article_type}, flach auf weißem Hintergrund liegend. "
        f"Merkmale: {desc_clean} Katalogstil, kein menschliches Modell, gleichmäßige Beleuchtung, saubere Produktdarstellung."
    )

def simple_translation_prompt(name, desc_de, model_descr):
    desc_clean = clean_description(desc_de)
    model_descr_clean = model_descr.replace(",", ", ").replace("  ", " ")
    return (
        f"Studioaufnahme von '{name}' mit Eigenschaften wie {model_descr_clean}. "
        f"{desc_clean} Weißer Hintergrund, keine Personen, Katalogdarstellung."
    )

def nlp_yake_prompt(desc_de, article_type):
    desc_clean = clean_description(desc_de)
    keywords = [kw for kw, _ in kw_extractor.extract_keywords(desc_clean)]
    top_keywords = ", ".join(keywords[:3])
    return (
        f"Studiofoto eines {article_type} im flachen Winkel mit den Hauptmerkmalen: {top_keywords}. "
        "Weißer Hintergrund, Katalogstil, keine Menschen oder Ablenkungen."
    )

# === Generate Prompts ===
prompt_data = []
for idx, row in df.iterrows():
    name = row['name']
    desc_de = row['description']
    article_type = row['model_name']
    model_descr = row['model_descr']

    prompt_manual = manual_template_prompt(article_type, desc_de)
    prompt_simple = simple_translation_prompt(name, desc_de, model_descr)
    prompt_nlp = nlp_yake_prompt(desc_de, article_type)

    prompt_data.append({
        "Beschreibung_Deutsch": clean_description(desc_de),
        "Prompt_Manuell": prompt_manual,
        "Prompt_Einfach": prompt_simple,
        "Prompt_NLP": prompt_nlp,
        "Negativ_Prompt": negative_prompt_text
    })

# === Create Final DataFrame with Original Headings + Prompts ===
df_prompts = pd.DataFrame(prompt_data)
df_final = df[['name', 'description', 'model_name', 'model_descr']].copy()
df_final["Beschreibung_Deutsch"] = df_prompts["Beschreibung_Deutsch"]
df_final["Prompt_Manuell"] = df_prompts["Prompt_Manuell"]
df_final["Prompt_Einfach"] = df_prompts["Prompt_Einfach"]
df_final["Prompt_NLP"] = df_prompts["Prompt_NLP"]
df_final["Negativ_Prompt"] = df_prompts["Negativ_Prompt"]

# === Save to Excel ===
output_path = "engineered_prompts_de_cleaned.xlsx"
df_final.to_excel(output_path, index=False)
print(f"✅ Alle bereinigten deutschen Prompts wurden gespeichert unter {output_path}")

# === Show examples ===
print("\n=== BEISPIELPROMPTS (erste 5 Einträge) ===")
for i in range(min(5, len(df_final))):
    row = df_final.iloc[i]
    print(f"\n🟡 Name: {row['name']}")
    print(f"📦 Artikeltyp: {row['model_name']}")
    print(f"📝 Beschreibung: {row['Beschreibung_Deutsch']}")
    print(f"🔹 Manueller Prompt:\n   {row['Prompt_Manuell']}")
    print(f"🔹 Einfacher Prompt:\n   {row['Prompt_Einfach']}")
    print(f"🔹 NLP Prompt (YAKE):\n   {row['Prompt_NLP']}")
    print(f"❌ Negativ Prompt:\n   {row['Negativ_Prompt']}")
