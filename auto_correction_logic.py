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

def max_f(candidates, word):
    # if (len(candidates) == 0): return ""
    # return min(candidates, key=lambda w: (edit_distance(word, w, substitution_cost=2, transpositions=True), -word_counter[w]))
    # return min(candidates, key=lambda w: (lev(word, w), -word_counter[w]))
    if not candidates:
        return []

    scored = [
        (
            edit_distance(word, w, substitution_cost=2, transpositions=True),
            -word_counter[w],
            w
        )
        for w in candidates
    ]

    # best_score = min(scored)[:2]  # (distance, -freq)
    # print(sorted(scored))
    # return [w for d, f, w in scored if (d, f) == best_score]
    return sorted(scored)


sounds = dict()

word_tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]


for i in word_tokens:
    try:
        sounds.setdefault(soundex(i), set()).add(i.lower())
    except Exception:
        pass

# print(max_f(sounds.get(soundex("expectaon"), set()), "expectaon"))
# print(sounds.get(soundex("expectaon"), set()), "expectaon")

metasounds = dict()

for i in word_tokens:
    try:
        metasounds.setdefault(metaphone(i), set()).add(i.lower())
    except Exception:
        pass
# print(max_f(metasounds.get(metaphone("expectaon"), set()), "expectaon"))
# print(metasounds)
# given = input().strip()
# print(max_f([i for a, b, i in max_f(metasounds.get(metaphone(given), set()), given)] + [i for a, b, i in max_f(sounds.get(soundex(given), set()), given)], given))
# Code for taking in Any amount of values given by the user:

print("***This is a REPL (please press Enter to exit)***")
while 1:
    given = input("> ").strip()
    if (len(given) == 0): 
        print("Exiting...")
        break
    candidates = sorted(list(set(max_f([i[-1] for i in max_f(metasounds.get(metaphone(given), set()), given)] + [i[-1] for i in max_f(sounds.get(soundex(given), set()), given)], given))))
    
    print(f"Top 3 matches: \nmatches:")
    for i in range(3):
        print(candidates[i][-1], end=" ")
    print()
