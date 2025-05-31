import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
from collections import defaultdict
from langdetect import detect
import json
import string
import unicodedata
import re

st.set_page_config(page_title="🏛 Feature Extractor", layout="centered")
st.title("🏛 Historical Building Feature Extractor (FR/EN)")

# --- Text normalization ---
def normalize_text(text):
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    punctuation = string.punctuation.replace("'", "")
    translator = str.maketrans('', '', punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFC', text)
    return text

# --- Lemmatization + cleaning ---
def preprocess_with_lemma(text, nlp):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)

# --- Load spaCy models first (needed for lemmatizing keywords) ---
@st.cache_resource
def load_models():
    return {
        "en": spacy.load("en_core_web_sm"),
        "fr": spacy.load("fr_core_news_sm")
    }

NLP_MODELS = load_models()

# --- Load and normalize+lemmatize keywords ---
@st.cache_resource
def load_keywords_normalized():
    with open("keywords.json", "r", encoding="utf-8") as f:
        raw_keywords = json.load(f)

    normalized = {}
    for lang, categories in raw_keywords.items():
        normalized[lang] = {}
        nlp = NLP_MODELS[lang]
        for category, kw_list in categories.items():
            normalized[lang][category] = [
                preprocess_with_lemma(normalize_text(kw), nlp)
                for kw in kw_list
            ]
    return normalized

KEYWORDS = load_keywords_normalized()

# --- Create matcher ---
def create_matcher(nlp, keywords):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for category, kw_list in keywords.items():
        patterns = [nlp.make_doc(kw) for kw in kw_list]
        matcher.add(category, patterns)
    return matcher

# --- Feature extraction ---
def extract_features(text, lang):
    nlp = NLP_MODELS[lang]
    matcher = create_matcher(nlp, KEYWORDS[lang])

    # Clean and lemmatize text for matching
    normalized_text = normalize_text(text)
    lemmatized_text = preprocess_with_lemma(normalized_text, nlp)
    match_doc = nlp(lemmatized_text)

    # Original doc for named entity extraction
    original_doc = nlp(text)

    features = defaultdict(list)
    matches = matcher(match_doc)
    matched_spans = defaultdict(set)

    for match_id, start, end in matches:
        category = nlp.vocab.strings[match_id]
        span_text = match_doc[start:end].text
        matched_spans[category].add(span_text)

    for category, spans in matched_spans.items():
        features[category] = sorted(spans)

    named_entities = [
        ent.text for ent in original_doc.ents
        if ent.label_ in ["LOC", "PER", "MISC", "ORG", "DATE", "WORK_OF_ART"]
    ]
    features["Entités nommées" if lang == "fr" else "Named Entities"] = named_entities

    return dict(features)

# --- Streamlit UI ---
text_input = st.text_area("Paste or write a paragraph in French or English:")

if st.button("Extract Features"):
    if text_input.strip():
        with st.spinner("🔍 Detecting language and analyzing..."):
            try:
                lang = detect(text_input)
                lang = "fr" if lang.startswith("fr") else "en"
                st.success(f"✅ Language detected: {'Français' if lang == 'fr' else 'English'}")

                results = extract_features(text_input, lang)

                for category, items in results.items():
                    st.subheader(category)
                    if items:
                        st.markdown(", ".join(items))
                    else:
                        st.markdown("Aucun élément trouvé." if lang == "fr" else "None found.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    else:
        st.warning("⚠ Please enter a paragraph to analyze.")