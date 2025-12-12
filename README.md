# Text Autocorrection System
A simple text autocorrection system should be able to correct basic words typed by the user, identify keyboard proximity errors, provide with more than one suggestion and more importantly -- be fast.

My aim for this project is to get to know:
- More efficient and real world uses of Data Structures.
- Implementation of AI&ML in real-life scenarios.

## The current Status of this project
This is the first commit I have done in this project, I have made a basic implementation of a text-autocorrection system by utilizing various libraries in Python. 

### Implementation structure:
- I have used `nltk` library to tokenize words from `big.txt` corpus.
- Then, used `collections` library to make a frequency map.
- With the help of `soundex` and `metaphone`, which come with the `jellyfish` library, I made a basic work-around for a text-autocorrection system.
- Also, with the help of `edit-distance` which measures the **Levenshtein distance** between two words, I optimized the results.

### To Do:
- Add Keyboard Proximity error resolution
- Make the result generation much faster using Data Structres and pre-computation.
- Integrate AI&ML.

*Planning on completing this project within a month*

