# -*- coding: utf-8 -*-
"""
Refined English Prompt Engineering Script
"""

import os, glob, re
import pandas as pd
import yake
import torch
from deep_translator import GoogleTranslator  # ✅ REPLACEMENT FOR translate.Translator

# ── SETTINGS ────────────────────────────────────────────────────────────────
FOLDER         = r"E:/CaseStudy/NKD Image generation datasets-20250505"  # path with .xlsx files
OUTPUT_CSV     = "engineered_prompts_adv_all.xlsx"
MAX_WORDS      = 50
# ────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
negative_prompt = "human, face, model, pose, mannequin, skin, hands, watermark, text, cropped"

brand_map = {
    "Mickey Mouse": "cartoon mouse",
    "Disney": "cartoon character",
    "Marvel": "comic graphic",
    "Star Wars": "sci-fi design"
}
fluff_words = {"comfortable", "quality", "feeling", "wearing", "pleasant", "optimal", "offers", "ensures"}

# ── UTILITIES ───────────────────────────────────────────────────────────────
def translate_text(text: str) -> str:
    try:
        return GoogleTranslator(source='de', target='en').translate(text)
    except Exception as e:
        print("⚠️ Translation failed:", e)
        return "TRANSLATION_FAILED"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for brand, generic in brand_map.items():
        text = text.replace(brand, generic)
    words = [w for w in text.split() if w.lower() not in fluff_words]
    return " ".join(dict.fromkeys(words)).strip()

def extract_visual_summary(text: str) -> str:
    patterns = ("neck", "sleeve", "pattern", "print", "fabric", "design", "color", "cut", "shape", "material", "texture")
    sentences = [s.strip() for s in text.split('.') if any(p in s.lower() for p in patterns)]
    return ", ".join(sentences[:2]) if sentences else ""

def build_sdxl_prompt(article_type: str, keywords: str, summary: str) -> str:
    prompt = (
        f"A {article_type.lower()} featuring {keywords}, {summary}, "
        "studio lighting, white background, product photo, catalog style, no human model"
    )
    words = prompt.split()
    return " ".join(words[:MAX_WORDS]) if len(words) > MAX_WORDS else prompt

# ── PROCESS ALL FILES ───────────────────────────────────────────────────────
rows = []
for file in glob.glob(os.path.join(FOLDER, "*.xlsx")):
    print(f"🔄 Reading {os.path.basename(file)} …")
    df = pd.read_excel(file).dropna(subset=['name', 'description', 'model_name', 'model_descr'])

    for _, r in df.iterrows():
        name, desc_de, article_type, model_descr = r[['name', 'description', 'model_name', 'model_descr']]

        desc_en = translate_text(desc_de)
        if desc_en == "TRANSLATION_FAILED":
            rows.append({
                "Source_File": os.path.basename(file),
                "Name": name,
                "Article_Type": article_type,
                "Model_Descr": model_descr,
                "Description_German": desc_de,
                "Description_English": "",
                "Keywords": "",
                "Summary": "",
                "Prompt_Final": "",
                "Negative_Prompt": negative_prompt
            })
            continue

        desc_en_clean = clean_text(desc_en)
        keywords = ", ".join(kw for kw, _ in kw_extractor.extract_keywords(desc_en_clean))
        summary = extract_visual_summary(desc_en_clean)
        final_prompt = build_sdxl_prompt(clean_text(article_type), clean_text(keywords), clean_text(summary))

        rows.append({
            "Source_File": os.path.basename(file),
            "Name": name,
            "Article_Type": article_type,
            "Model_Descr": model_descr,
            "Description_German": desc_de,
            "Description_English": desc_en,
            "Keywords": keywords,
            "Summary": summary,
            "Prompt_Final": final_prompt,
            "Negative_Prompt": negative_prompt
        })

# ── SAVE OUTPUT ─────────────────────────────────────────────────────────────
df_out = pd.DataFrame(rows)
df_out.to_excel(OUTPUT_CSV, index=False)
print(f"\n✅ Saved {len(df_out)} rows to {OUTPUT_CSV}")

# ── SAMPLE OUTPUT ───────────────────────────────────────────────────────────
print("\n=== SAMPLE OUTPUT ===")
for i, row in df_out.head(3).iterrows():
    print(f"\n🔹 ({i+1}) {row['Name']}")
    print("   EN desc :", row['Description_English'][:120] + "...")
    print("   Keywords:", row['Keywords'])
    print("   Summary :", row['Summary'])
    print("   Prompt  :", row['Prompt_Final'][:140] + ("..." if len(row['Prompt_Final']) > 140 else ""))
