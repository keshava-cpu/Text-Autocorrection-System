# Text Autocorrection System
A simple text autocorrection system should be able to correct basic words typed by the user, identify keyboard proximity errors, provide with more than one suggestion and more importantly -- be fast.

My aim for this project is to get to know:
- More efficient and real world uses of Data Structures.
- Implementation of AI&ML in real-life scenarios.

## The current Status of this project
The project has come a long way, I have added the basic version of the project. Then, I build upon that to finalize on an updated version.
The Current Project has these features:
- candidate generation (edits, phonetic, keyboard proximity, affixes)
- multi-factor scoring (frequency, edit distance, positional similarity, phonetic)
- normalization for contractions, repeated chars and leet/substitutions

### Implementation structure:
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

### To Do:
- Make the result generation much faster using Data Structres and pre-computation.
- Integrate AI&ML.

