import pandas as pd
import os
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def normalize_text(text: str) -> str:
    """Normalize and clean input text for grouping."""
    text = text.strip()

    text = re.sub(r"\b(ad[- ]?hoc)\b", "adhoc", text, flags=re.IGNORECASE)
    # unify spaces and punctuation
    text = re.sub(r"[-_.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.upper().strip()

    # map synonyms to canonical form
    mapping = {
        "ADHOC": "ADHOC",
        "AD HOC": "ADHOC",
        "AD-HOC": "ADHOC",
        "AD HOCS": "ADHOC",
        "AD-HOCS": "ADHOC",
        "ANALYTICAL": "ANALYTIC",
        "ANALYTICS": "ANALYTIC",
        "APPLICATIONS": "APPLICATION",
        "MODELS": "MODEL",
        "COMPANIES": "COMPANY",
        "PRODUCTIVITY": "PRODUCTS",
        "PRODUCTION": "PRODUCTS",
        "PRODUCTIONS": "PRODUCTS",
    }

    # apply word-by-word mapping
    words = text.split()
    cleaned = [mapping.get(w, w) for w in words]
    text = " ".join(cleaned).strip()

    # manually unify variants
    replacements = {
        "ADHOC QUERYING": "ADHOC QUERIES",
        "ADHOC QUERY": "ADHOC QUERIES",
        "ADHOC QUERIES": "ADHOC QUERIES",
        "AGRICULTURAL PRODUCTS": "AGRICULTURAL PRODUCTS",
        "AIRLINE COMPANY": "AIRLINE COMPANY",
        "AIRLINES COMPANY": "AIRLINE COMPANY",
        "AIRLINE COMPANIES": "AIRLINE COMPANY",
        "AIRLINES": "AIRLINES",
        "ANALYTIC MODEL": "ANALYTICS MODEL",
        "ANALYTICAL MODEL": "ANALYTICS MODEL",
        "ANALYTICAL MODELS": "ANALYTICS MODEL",
        "ANALYTIC MODELS": "ANALYTICS MODEL",
        "ANALYTIC MODELING": "ANALYTICS MODEL",
        "ANALYTICS MODEL": "ANALYTICS MODEL",
        "ANALYTICS APPLICATION": "Analytics Application",
        "ANALYTIC APPLICATION": "Analytics Application",
        "ANALYTICAL APPLICATION": "Analytics Application",
        "ANALYTICAL APPLICATIONS": "Analytics Application",
        "ANALYTICS APPLICATIONS": "Analytics Application",
    }
    if text in replacements:
        text = replacements[text]
    return text

def make_key(text: str) -> str:
    """Generate a stemmed lowercase key."""
    text = re.sub(r"\b(ad[- ]?hoc)\b", "adhoc", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z ]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(stemmer.stem(w) for w in words)

def main(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df["cleaned_text"] = df["raw_text"].apply(normalize_text)
    df["key"] = df["raw_text"].apply(make_key)

    # Save test.csv
    test_path = os.path.join(os.path.dirname(output_path), "test.csv")
    df[["key"]].to_csv(test_path, index=False)

    # Save output.txt (as CSV)
    df[["cleaned_text"]].to_csv(output_path, index=False)
