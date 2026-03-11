import os, glob, re, pandas as pd, yake
from deep_translator import GoogleTranslator

# ========= SETTINGS =========
# Folder that contains all of your source .xlsx files
FOLDER = r"E:/CaseStudy/NKD Image generation datasets-20250505"   
OUTPUT_XLSX = "engineered_prompts_all.xlsx"                       # final combined file
# (use .csv instead if you want CSV)

# ========= CONSTANTS =========
kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
negative_prompt_text = (
    "human, person, face, head, arms, legs, hands, mannequin, "
)

# ========= HELPER FUNCTIONS =========
def translate_text(text):
    try:
        return GoogleTranslator(source='de', target='en').translate(text)
    except Exception:
        return "TRANSLATION_FAILED"

def clean_description(text):
    if not isinstance(text, str):
        return ""
    lines = text.strip().split("\n")
    paragraph = []
    for line in lines:
        t = line.strip()
        if t.startswith(("-", "–", "•", "\t")) or t == "":
            break
        paragraph.append(t)
    cleaned = " ".join(paragraph)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip().rstrip(".") + "."

def manual_template_prompt(article_type, desc_en):
    return (
        f"Studio-quality image of a single {article_type}, laid flat on a clean white background. "
        f"Features include: {clean_description(desc_en)} Catalog-style photography. "
        "No people, no men, no women, no human models. Even lighting, sharp focus, clean product presentation."
    )

def simple_translation_prompt(name, desc_en, model_descr):
    model_descr_clean = model_descr.replace(",", ", ").replace("  ", " ")
    return (
        f"Studio-quality product photo of '{name}' showing features such as {model_descr_clean}. "
        f"{clean_description(desc_en)} Plain white background. "
        "No humans, no male or female figures. Professional catalog layout."
    )

def nlp_yake_prompt(desc_en, article_type):
    keywords = [kw for kw, _ in kw_extractor.extract_keywords(clean_description(desc_en))]
    top_keywords = ", ".join(keywords[:3])
    return (
        f"High-quality studio photo of a {article_type} photographed at a soft angle, "
        f"showing key features: {top_keywords}. "
        "White background, catalog-style shot. No men, no women, no human models or figures present."
    )

# ========= MAIN LOOP =========
prompt_rows = []

excel_files = glob.glob(os.path.join(FOLDER, "*.xlsx"))
for path in excel_files:
    df = pd.read_excel(path).dropna(subset=['name', 'description', 'model_name', 'model_descr'])
    for _, row in df.iterrows():
        name, desc_de, article_type, model_descr = row[['name', 'description', 'model_name', 'model_descr']]
        desc_en = translate_text(desc_de)

        if desc_en == "TRANSLATION_FAILED":
            # Skip or record an empty set; here we skip
            continue

        prompt_rows.append({
            "Source_File": os.path.basename(path),
            "Name": name,
            "Article_Type": article_type,
            "Model_Descr": model_descr,
            "Description_German": desc_de,
            "Description_English": desc_en,
            "Prompt_Manual":  manual_template_prompt(article_type, desc_en),
            "Prompt_Simple":  simple_translation_prompt(name, desc_en, model_descr),
            "Prompt_NLP":     nlp_yake_prompt(desc_en, article_type),
            "Negative_Prompt": negative_prompt_text
        })

# ========= SAVE =========
df_all = pd.DataFrame(prompt_rows)

# >>> choose one of the two <<<

df_all.to_excel(OUTPUT_XLSX, index=False)      

print(f" Combined prompts saved to {OUTPUT_XLSX} with {len(df_all)} rows.")

