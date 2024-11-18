import re

import nltk
from nltk.stem.snowball import SnowballStemmer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


TRASH_REGEX = re.compile(r"[  ’‘'\*\~\_]")
LIRA_PRICE_REGEX = re.compile(r"([0-9]+[.,]?[0-9]*)[  ]*(TL|₺)")
DOLLAR_PRICE_REGEX = re.compile(r"([0-9]+[.,]?[0-9]*)[  ]*(\$)")

emoji_pattern = re.compile(
    "["
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"
    "]+"
)


def clean_text(text: str):
    text = re.sub(r"[  ’‘'\*\~\_]", r" ", text)
    text = emoji_pattern.sub(r"", text)
    text = re.sub("[ ]+", r" ", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line)

    return stemming(text)


def stemming(text: str):
    stemmer = SnowballStemmer("russian")
    tokens = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_words)
