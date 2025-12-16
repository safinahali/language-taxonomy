import spacy
from PyPDF2 import PdfReader

# -------------------------------
# CONFIG
# -------------------------------
TARGET_NOUNS = {
    "ai",
    "artificial intelligence",
    "machine",
    "agent",
    "robot",
    "robots",
    "tutor",
    "AI"
}

PDF_PATH = "input.pdf"

# -------------------------------
# LOAD NLP MODEL
# -------------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        if page.extract_text():
            text.append(page.extract_text())
    return "\n".join(text)

# -------------------------------
# FIND NOUN → VERB PAIRS
# -------------------------------
def find_noun_verb_pairs(text):
    doc = nlp(text)
    results = []

    for sent in doc.sents:
        for token in sent:
            token_text = token.text.lower()

            # Handle multi-word noun: "Artificial Intelligence"
            if token_text == "artificial" and token.nbor(1).text.lower() == "intelligence":
                noun = "Artificial Intelligence"
                head = token.head
            elif token_text in TARGET_NOUNS:
                noun = token.text
                head = token.head
            else:
                continue

            # Check if noun is subject of a verb
            if token.dep_ in {"nsubj", "nsubjpass"} and head.pos_ == "VERB":
                results.append((noun, head.text))

    return results

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    text = extract_text_from_pdf(PDF_PATH)
    pairs = find_noun_verb_pairs(text)

    print("\nNoun → Verb pairs found:\n")
    for noun, verb in pairs:
        print(f"{noun} → {verb}")
