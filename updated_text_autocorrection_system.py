"""
Smart Autocorrect Module
------------------------

Purpose
    Lightweight autocorrect/suggestion system built on a local corpus.
    Provides:
      - candidate generation (edits, phonetic, keyboard proximity, affixes)
      - multi-factor scoring (frequency, edit distance, positional similarity, phonetic)
      - normalization for contractions, repeated chars and leet/substitutions

Primary functions
    - correct(word) -> best single correction (string)
    - suggest(word, n=3) -> top-n suggestions (list[str])

How it works (high level)
    1. Load a corpus (big.txt) and build unigram frequency counts.
    2. Generate candidate corrections using:
         - edits1 (single-edit variants) and second-order edits
         - phonetic matching (soundex/metaphone)
         - keyboard proximity swaps
         - prefix/suffix heuristics
         - normalization transforms (numbers to letters, repeated chars, contractions)
    3. Score candidates by combining:
         - normalized unigram frequency
         - edit distance (inverse)
         - sequence similarity (difflib)
         - positional character similarity
         - phonetic match indicator
    4. Return top candidates sorted by combined score.

Dependencies
    - Python 3.x
    - nltk (tokenizers, edit_distance)
    - jellyfish (soundex/metaphone)
    - Optional: inflect (not required for the core algorithm as-is)

Install
    pip install nltk jellyfish

Configuration / tuning
    - edits1(max_edit, min_similarity): tune to control how permissive the single-edit filter is.
    - score_candidate weights: adjust relative importance of frequency vs. form-similarity vs. phonetic.
    - QWERTY_NEIGHBORS: customize to reflect different keyboard layouts.
    - Corpus: using a large well-formatted big.txt improves coverage and ranking reliability.

Performance notes
    - Building phonetic matches and keyboard candidates scans the vocabulary; for large corpora precompute indexes (delete dictionary or phonetic index) to speed runtime suggestions.
    - edits1 currently builds and filters many variants; consider SymSpell-style delete dictionary if latency is important.
    - Memory: storing the corpus as tokens and Counter may consume significant RAM for very large corpora.

Examples
    >>> from test2 import suggest, correct
    >>> suggest("speling", n=3)
    ['spelling', 'spieling', 'sealing']
    >>> correct("recieve")
    'receive'

Security / Limitations
    - This is an offline statistical/heuristic approach; does not use large language models or deep context.
    - For multi-word or strong contextual suggestions, extend with bigram/trigram probabilities or a contextual model.

"""

import string
import re
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from difflib import SequenceMatcher
from jellyfish import soundex, metaphone  # pip install jellyfish
import functools

# ------------------ Load Corpus ------------------
text = ''
# Use a path relative to this script so packaged apps can include big.txt alongside the exe/wheel.
corpus_path = Path(__file__).parent / "big.txt"
with open(str(corpus_path), "r", encoding="utf8") as f:
    text = f.read()

word_tokens = word_tokenize(text)
word_count = Counter(word_tokens)

# build small helper structures once
vocab = set(word_count.keys())
max_freq = max(word_count.values()) if word_count else 1

# phonetic indexes for fast lookup (precompute)
phonetic_index_soundex = defaultdict(set)
phonetic_index_metaphone = defaultdict(set)
for w in vocab:
    try:
        phonetic_index_soundex[soundex(w)].add(w)
    except Exception:
        pass
    try:
        phonetic_index_metaphone[metaphone(w)].add(w)
    except Exception:
        pass

# small util: generate keyboard neighbor variants of token (cheap at query time)
def keyboard_neighbor_variants(token):
    variants = set()
    for i, c in enumerate(token):
        if c in QWERTY_NEIGHBORS:
            for neighbor in QWERTY_NEIGHBORS[c]:
                variants.add(token[:i] + neighbor + token[i+1:])
    return variants

# Cached per-token suggestion generator to avoid repeated heavy computations while typing
@functools.lru_cache(maxsize=8192)
def suggest_token_cached(token, n=5, candidate_cap=200):
    """
    Return top-n candidate suggestions for a single token.
    Cached to make live completions snappy.
    """
    t = token.lower().strip()
    if not t:
        return []

    # Fast exact hit
    if t in vocab:
        return [t]

    candidates = set()

    # 1) phonetic prefilter (fast) - union limited to reasonable counts
    s_code = soundex(t)
    m_code = metaphone(t)
    if s_code in phonetic_index_soundex:
        candidates |= set(list(phonetic_index_soundex[s_code])[:100])
    if m_code in phonetic_index_metaphone:
        candidates |= set(list(phonetic_index_metaphone[m_code])[:100])

    # 2) keyboard proximity variants (cheap)
    kb_vars = keyboard_neighbor_variants(t)
    candidates |= set(w for w in kb_vars if w in vocab)

    # 3) single-edit known words (only if still small)
    if len(candidates) < 20:
        edit1_known = known(edits1(t, max_edit=1, min_similarity=0.5))
        candidates |= set(edit1_known)

    # 4) second-order edits only if needed (expensive)
    if not candidates:
        # generate second-order but limit growth by filtering lengths
        e1 = edits1(t, max_edit=1, min_similarity=0.5)
        e2_candidates = set()
        for e in e1:
            e2_candidates |= set(known(edits1(e, max_edit=1, min_similarity=0.5)))
            if len(e2_candidates) > 200:
                break
        candidates |= e2_candidates

    # Cap candidates to keep scoring fast
    if len(candidates) > candidate_cap:
        # keep top by unigram freq as prefilter
        candidates = set(sorted(candidates, key=lambda w: word_count.get(w,0), reverse=True)[:candidate_cap])

    # Score and return top-n
    scored = []
    for cand in candidates:
        freq = word_count.get(cand, 0)
        try:
            sc = score_candidate(t, cand, freq)
        except Exception:
            sc = 0.0
        scored.append((sc, cand))
    scored.sort(reverse=True)
    return [cand for _, cand in scored[:n]]

# ------------------ Edit Distance Functions ------------------
def edits1(word, max_edit=1, min_similarity=0.6):
    """
    Generate single-edit candidate strings for `word`, filtered by two heuristics.

    Parameters
    ----------
    word : str
        Input token to generate edited variants for.
    max_edit : int, default=1
        Maximum allowed Levenshtein edit distance between the input word and a candidate.
        Keep low (1 or 2) to limit noisy candidates.
    min_similarity : float, default=0.6
        Minimum SequenceMatcher ratio between input and candidate to allow.
        This filters out variants that are too dissimilar lexically.

    Returns
    -------
    set[str]
        A set of candidate strings that satisfy both edit distance and similarity filters.

    Notes
    -----
    - Produces deletes, transposes, replaces and inserts.
    - Filtering helps avoid vast candidate explosion, but may filter legitimate multi-character transpositions;
      tune thresholds when needed.
    """
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    
    candidates = set(deletes + transposes + replaces + inserts)
    # Filter candidates by edit distance and sequence similarity
    filtered = set()
    for cand in candidates:
        ed = edit_distance(word, cand)
        sim = SequenceMatcher(None, word, cand).ratio()
        if ed <= max_edit and sim >= min_similarity:
            filtered.add(cand)
    return filtered

def known(words):
    """Return the subset of words that appear in the corpus"""
    return list(set(w for w in words if w in word_count))

def char_position_similarity(s1, s2):
    """
    Compute a positional similarity score between two strings.

    Purpose
    -------
    Measures how aligned the positions of shared characters are between s1 and s2.
    Useful for detecting jumbled characters that still preserve positional hints (e.g., "adrse" vs "address").

    Returns
    -------
    float
        Score in [0.0, 1.0] where higher means characters appear in similar relative positions.
    """
    # Convert strings to character sets
    s1_chars = set(s1)
    s2_chars = set(s2)
    
    # Calculate character overlap
    common_chars = s1_chars & s2_chars
    if not common_chars:
        return 0.0
    
    # Compare positions of common characters
    positions_score = 0
    len_s1, len_s2 = len(s1), len(s2)
    max_len = max(len_s1, len_s2)
    
    for char in common_chars:
        # Get all positions of the character in both strings
        pos_s1 = [i/len_s1 for i, c in enumerate(s1) if c == char]
        pos_s2 = [i/len_s2 for i, c in enumerate(s2) if c == char]
        
        # Compare closest positions
        min_diff = min(abs(p1-p2) for p1 in pos_s1 for p2 in pos_s2)
        positions_score += 1 - min_diff
    
    return positions_score / max(len(s1_chars), len(s2_chars))

def phonetic_candidates(word, vocab):
    """
    Return words from `vocab` that share a phonetic encoding with `word`.

    Parameters
    ----------
    word : str
        Input token.
    vocab : iterable[str]
        Vocabulary to search (e.g., word_count.keys()).

    Returns
    -------
    set[str]
        Words with matching Soundex or Metaphone encodings and reasonable length proximity.

    Notes
    -----
    - Phonetic matching helps catch transpositions or severe misspellings that sound like the target.
    - Consider precomputing a phonetic index for large vocabularies.
    """
    word_sound = soundex(word)
    word_metaphone = metaphone(word)
    matches = set()
    
    for v in vocab:
        if len(v) >= len(word)-2 and len(v) <= len(word)+2:  # Length filter
            if soundex(v) == word_sound or metaphone(v) == word_metaphone:
                matches.add(v)
    return matches

def score_candidate(word, candidate, freq):
    """
    Compute a combined score for ranking a candidate correction.

    Components
    ----------
    - seq_sim: SequenceMatcher ratio (0..1)
    - edit distance: converted to inverse score 1/(1+ed)
    - pos_sim: char positional similarity (0..1)
    - sound_sim: binary phonetic match (0 or 1)
    - freq: unigram frequency (normalized by max frequency in corpus)

    The final score is a weighted sum. Adjust weights to emphasize frequency vs. form similarity.

    Parameters
    ----------
    word : str
    candidate : str
    freq : int
        Raw frequency count for `candidate` in the corpus.

    Returns
    -------
    float
        Combined ranking score (higher is better).
    """
    ed = edit_distance(word, candidate)
    seq_sim = SequenceMatcher(None, word, candidate).ratio()
    pos_sim = char_position_similarity(word, candidate)
    sound_sim = float(soundex(word) == soundex(candidate))
    
    # Combine scores with weights
    score = (
        0.4 * seq_sim +           # Sequence similarity
        0.3 * (1/(1+ed)) +        # Edit distance (inverse)
        0.2 * pos_sim +           # Position similarity
        0.1 * sound_sim +         # Phonetic similarity
        0.3 * (freq/max(word_count.values()))  # Frequency (normalized)
    )
    return score

def normalize_word(word):
    """
    Normalize the input token before candidate generation.

    Steps performed
    1. Lowercase the token (internal processing).
    2. Replace common contractions (simple mapping).
    3. Collapse excessive repeated characters: 'helllllo' -> 'helllo' (two repeats).
    4. Substitute common leetspeak numbers to letters (e.g., 3->e).

    Returns the normalized token. If contraction map expands to a phrase, the returned value may include spaces.
    """
    # Store original capitalization
    case_pattern = [c.isupper() for c in word]
    
    # Convert to lowercase for processing
    word = word.lower()
    
    # Handle contractions
    contractions = {
        "cant": "cannot", "dont": "do not", "doesnt": "does not",
        "isnt": "is not", "wouldnt": "would not", "shouldnt": "should not"
    }
    if word in contractions:
        return contractions[word]
    
    # Remove repeated characters (more than 2 times)
    word = re.sub(r'(.)\1{2,}', r'\1\1', word)
    
    # Handle number substitutions
    number_map = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', 
                 '7': 't', '8': 'ate', '9': 'g'}
    for num, letter in number_map.items():
        word = word.replace(num, letter)
    
    return word

QWERTY_NEIGHBORS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs', 'e': 'rdsw3',
    'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'uko8', 'j': 'uikmnh',
    'k': 'iolmj', 'l': 'pokm', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp9',
    'p': 'ol0', 'q': '12wa', 'r': '45tfe', 's': 'wedxza', 't': '56ygfr',
    'u': '78ijhy', 'v': 'cfgb', 'w': '23esa', 'x': 'zsdc', 'y': '67uhgt',
    'z': 'asx'
}

def keyboard_proximity_candidates(word):
    """Generate candidates based on keyboard proximity"""
    candidates = set()
    for i, c in enumerate(word):
        if c in QWERTY_NEIGHBORS:
            for neighbor in QWERTY_NEIGHBORS[c]:
                candidates.add(word[:i] + neighbor + word[i+1:])
    return candidates

def handle_prefixes_suffixes(word):
    """Handle common prefix/suffix errors"""
    common_prefixes = ['un', 'in', 'dis', 're', 'pre']
    common_suffixes = ['ing', 'ed', 'ly', 'tion', 'ment']
    
    candidates = set()
    # Try removing/adding common prefixes
    for prefix in common_prefixes:
        if word.startswith(prefix):
            candidates.add(word[len(prefix):])
        else:
            candidates.add(prefix + word)
    
    # Try removing/adding common suffixes
    for suffix in common_suffixes:
        if word.endswith(suffix):
            candidates.add(word[:-len(suffix)])
        else:
            candidates.add(word + suffix)
    
    return candidates

def candidates(word):
    """
    Generate and combine candidate corrections for `word`.

    Candidate sources (in priority/fallback order)
    ---------------------------------------------
    - exact match in vocab
    - single-edit known words
    - double-edit known words (second-order edits)
    - keyboard-proximity corrections
    - prefix/suffix heuristics
    - phonetic matches
    - normalization result(s)

    Returns
    -------
    list[str]
        Candidate list (not yet ranked). Returned as a list for backward compatibility with calling code.

    Notes
    -----
    - The function converts internal sets to a list at the end.
    - For speed, consider returning a bounded number of candidates and precomputing indexes for large corpora.
    """
    # Normalize the word first
    normalized = normalize_word(word)
    if normalized != word:
        initial_candidates = {normalized}
    else:
        initial_candidates = set()
    
    # Get edit distance candidates and convert to set
    edit_candidates = set(
        known([normalized]) or
        known(edits1(normalized)) or
        known([e2 for e1 in edits1(normalized) 
              for e2 in edits1(e1) if len(e2) >= len(normalized)-1])
    )
    
    # Convert keyboard candidates to set
    keyboard_candidates = set(known(keyboard_proximity_candidates(normalized)))
    
    # Convert affix candidates to set
    affix_candidates = set(known(handle_prefixes_suffixes(normalized)))
    
    # Phonetic matches is already a set
    phonetic_matches = phonetic_candidates(normalized, word_count.keys())
    
    # Combine all candidates
    all_candidates = (edit_candidates | keyboard_candidates | 
                     affix_candidates | phonetic_matches | initial_candidates)
    
    return list(all_candidates) if all_candidates else [word]

# ------------------ Correction Functions ------------------
def correct(word):
    """
    Return the single best correction for `word`.

    This wraps suggest(word, n=1) and returns the first element.

    Behavior
    --------
    - If multiple candidates tie, ranking uses the score_candidate function.
    - If no reasonable candidate is found, returns the original word (fallback behavior).

    Example
    -------
    >>> correct("speling")
    'spelling'
    """
    return suggest(word, n=1)[0]

def suggest(word, n=3):
    """
    Return the top-n ranked suggestions for `word`.

    Parameters
    ----------
    word : str
    n : int, default=3
        Number of suggestions to return.

    Returns
    -------
    list[str]
        Sorted list of suggested corrections (highest scoring first).

    Example
    -------
    >>> suggest("recieve", n=3)
    ['receive', 'recipe', 'receiver']
    """
    cand = candidates(word)
    return sorted(cand, key=lambda w: score_candidate(word, w, word_count[w]), reverse=True)[:n]

# ------------------ CLI Entry Point ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Smart Autocorrect CLI")
    p.add_argument('words', nargs='*', help='Word(s) to correct or suggest for (space-separated).')
    p.add_argument('-f', '--file', type=str, help='Path to a file containing words (one per line).')
    p.add_argument('-n', '--num', type=int, default=3, help='Number of suggestions to return (suggest mode).')
    p.add_argument('--mode', choices=['suggest', 'correct'], default='suggest',
                   help='Operation mode: "suggest" returns top-n suggestions; "correct" returns best correction.')
    p.add_argument('-i', '--interactive', action='store_true', help='Start interactive REPL (ignore words/file).')
    p.add_argument('-l', '--live', action='store_true', help='Start live REPL with real-time suggestions (requires prompt_toolkit).')
    return p.parse_args()

def process_words(words, args):
    results = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if args.mode == 'correct':
            try:
                res = correct(w)
            except Exception:
                res = w
            results.append((w, res))
        else:
            try:
                res = suggest(w, n=args.num)
            except Exception:
                res = [w]
            results.append((w, res))
    return results

def repl(args):
    print("Smart Autocorrect REPL — type a word and press Enter (empty to quit).")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                break
            out = process_words([q], args)
            for inp, res in out:
                if args.mode == 'correct':
                    print(res)
                else:
                    print(", ".join(res))
    except (KeyboardInterrupt, EOFError):
        print("\nExiting REPL.")

def read_words_from_file(path):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return []
    with p.open("r", encoding="utf8") as fh:
        return [line.strip() for line in fh if line.strip()]

def live_repl(args):
    import importlib, importlib.util, traceback, sys
    # Try direct, version-compatible imports for prompt_toolkit API
    try:
        from prompt_toolkit.shortcuts import prompt
        from prompt_toolkit.application.current import get_app
        from prompt_toolkit.completion import Completer, Completion
    except Exception as exc:
        # Print concise diagnostic and fall back to simple REPL
        print("prompt_toolkit import failed:", exc, file=sys.stderr)
        print("Attempting diagnostic info...", file=sys.stderr)
        try:
            spec = importlib.util.find_spec('prompt_toolkit')
            print("  prompt_toolkit spec:", spec, file=sys.stderr)
            if spec and getattr(spec, "origin", None):
                print("  spec.origin:", spec.origin, file=sys.stderr)
        except Exception:
            pass
        print(f"Use the same Python used to run this script to install prompt_toolkit:", file=sys.stderr)
        print(f"  {sys.executable} -m pip install prompt_toolkit", file=sys.stderr)
        print("Falling back to the simple REPL (no live suggestions).", file=sys.stderr)
        repl(args)
        return

    class DynamicCompleter(Completer):
        def get_completions(self, document, complete_event):
            text_before = document.text_before_cursor
            last = text_before.split()[-1] if text_before.strip() else ''
            if not last:
                return
            # Use cached per-token suggestions for speed
            try:
                suggestions = suggest_token_cached(last, n=10)
            except Exception:
                suggestions = []
            for s in suggestions:
                yield Completion(s, start_position=-len(last))

    def bottom_toolbar():
        try:
            app = get_app()
            text = app.current_buffer.document.text
            last = text.split()[-1] if text.strip() else ''
            if not last:
                return ''
            sug = suggest_token_cached(last, n=5)
            return ' Suggestions: ' + ', '.join(sug)
        except Exception:
            return ''

    print("Live REPL — suggestions appear as you type. Enter empty line to quit.")
    while True:
        try:
            user = prompt('> ', completer=DynamicCompleter(), bottom_toolbar=bottom_toolbar)
            if not user.strip():
                break
            # On enter, show final suggestions for last word and continue
            last = user.split()[-1] if user.strip() else ''
            if last:
                print("Final suggestions:", ', '.join(suggest_token_cached(last, n=args.num)))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting live REPL.")
            break

def main():
    args = parse_args()

    if args.live:
        live_repl(args)
        return

    if args.interactive:
        repl(args)
        return

    words = []
    if args.file:
        words.extend(read_words_from_file(args.file))
    if args.words:
        words.extend(args.words)

    if not words:
        print("No words provided. Use --interactive or pass words or -f/--file.", file=sys.stderr)
        sys.exit(1)

    results = process_words(words, args)
    # Print results: for suggest show comma-separated suggestions, for correct show single correction
    for inp, res in results:
        if args.mode == 'correct':
            print(f"{inp} -> {res}")
        else:
            print(f"{inp} -> {', '.join(res)}")

if __name__ == "__main__":
    main()
