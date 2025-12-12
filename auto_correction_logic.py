# Lets start by importing the required corpus, then we will try to do some tokenization 
# and then some text analysis.

# We will be using Path from pathlib
from pathlib import Path
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from jellyfish import soundex, metaphone
from nltk.metrics import edit_distance


text = ""
corpus_path = Path(__file__).parent / "big.txt"

with open(corpus_path, "r") as f:
    text = f.read()
word_tokens = word_tokenize(text)
word_counter = Counter(word_tokens)

def max_f(i):
    return min(i, key=lambda w: (edit_distance("tre", w, substitution_cost=2, transpositions=True)))


sounds = dict()

for i in word_tokens:
    try:
        sounds.setdefault(soundex(i), set()).add(i.lower())
    except Exception:
        pass

print(max_f(sounds[soundex("tree")]))

metasounds = dict()

for i in word_tokens:
    try:
        metasounds.setdefault(metaphone(i), set()).add(i.lower())
    except Exception:
        pass
print(max_f(metasounds[metaphone("adres")]))