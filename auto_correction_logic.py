# Lets start by importing the required corpus, then we will try to do some tokenization 
# and then some text analysis.

# We will be using Path from pathlib
from pathlib import Path
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from jellyfish import soundex, metaphone
from nltk.metrics import edit_distance
from levenshtein_distance_calc import lev

text = ""
corpus_path = Path(__file__).parent / "big.txt"

with open(corpus_path, "r") as f:
    text = f.read()
word_tokens = word_tokenize(text)
word_counter = Counter(word_tokens)

def max_f(i, s):
    if (len(i) == 0): return ""
    return min(i, key=lambda w: (edit_distance(s, w, substitution_cost=2, transpositions=True), -word_counter[w]))
    # return min(i, key=lambda w: (lev(s, w), -word_counter[w]))


sounds = dict()

word_tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]


for i in word_tokens:
    try:
        sounds.setdefault(soundex(i), set()).add(i.lower())
    except Exception:
        pass

print(max_f(sounds.get(soundex("expectaon"), set()), "expectaon"))
print(sounds.get(soundex("expectaon"), set()), "expectaon")

metasounds = dict()

for i in word_tokens:
    try:
        metasounds.setdefault(metaphone(i), set()).add(i.lower())
    except Exception:
        pass
print(max_f(metasounds.get(metaphone("expectaon"), set()), "expectaon"))
# print(metasounds)
print(metasounds.get(metaphone("expectaon"), set()))