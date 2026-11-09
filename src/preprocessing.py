import re
import nltk
from nltk.corpus import stopwords
from razdel import tokenize
from pymorphy3 import MorphAnalyzer

nltk.download("stopwords")

russian_stopwords = set(
    stopwords.words("russian")
)

morph = MorphAnalyzer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(
        r"[^а-яёa-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    tokens = [
        token.text
        for token in tokenize(text)
    ]

    lemmas = []

    for token in tokens:

        if len(token) <= 2:
            continue

        if token in russian_stopwords:
            continue

        lemma = morph.parse(token)[0].normal_form

        lemmas.append(lemma)

    return " ".join(lemmas)