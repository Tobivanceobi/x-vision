import hashlib
import random
import sys
from itertools import product

from joblib import Parallel, delayed
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
# Seed for reproducibility
random.seed(0)

m_1 = "The lesser divinity_fudge pop_up."
m_2 = "The lesser extraneousness litter."

h = hashlib.blake2s(digest_size=6)
h.update(m_1.encode())
hash_value = h.hexdigest()
print(hash_value)
h = hashlib.blake2s(digest_size=6)
h.update(m_2.encode())
hash_value = h.hexdigest()
print(hash_value)

# Function to get words of a specific part of speech
def get_words(part_of_speech):
    return set([lemma.name() for synset in wordnet.all_synsets(part_of_speech) for lemma in synset.lemmas()])


# Get lists of nouns, verbs, and adjectives
nouns = get_words(wordnet.NOUN)
verbs = get_words(wordnet.VERB)
adjectives = get_words(wordnet.ADJ)

# Display the number of words in each category and a few examples
print(f"Number of nouns: {len(nouns)} - Example nouns: {list(nouns)[:5]}")
print(f"Number of verbs: {len(verbs)} - Example verbs: {list(verbs)[:5]}")
print(f"Number of adjectives: {len(adjectives)} - Example adjectives: {list(adjectives)[:5]}")

nouns = list(nouns)
verbs = list(verbs)
adjectives = list(adjectives)

random.shuffle(nouns)
random.shuffle(verbs)
random.shuffle(adjectives)

hashes = {}
iter_counter = 0
for a in adjectives:
    for n in nouns:
        for v in verbs:
            if iter_counter % 10000 == 0:
                print(iter_counter)
            sentence = f"The {a} {n} {v}."
            h = hashlib.blake2s(digest_size=7)
            h.update(sentence.encode())
            hash_value = h.hexdigest()
            if hash_value in hashes:
                print('found collision:')
                print(sentence)
                print(hashes[hash_value])
                sys.exit()
            if iter_counter % 23110000 == 0:
                hashes = {}
            hashes[hash_value] = sentence
            iter_counter += 1



