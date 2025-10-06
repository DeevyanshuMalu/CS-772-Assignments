import json
from collections import Counter
import Levenshtein as lev

BASE_CONSONANTS = {
    'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ch': 'च', 'chh': 'छ', 'j': 'ज', 'jh': 'झ',
    't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न', 'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ',
    'm': 'म', 'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व',
    'sh': 'श', 'shh': 'ष', 's': 'स', 'h': 'ह',
    'ṇ': 'ण', 'ṅ': 'ङ', 'ñ': 'ञ', 'ṭ': 'ट', 'ṭh': 'ठ', 'ḍ': 'ड', 'ḍh': 'ढ', 'ḷ': 'ळ'
}

VOWELS_INDEPENDENT = {
    'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ii': 'ई', 'u': 'उ', 'uu': 'ऊ',
    'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
    'ri': 'ऋ', 'rri': 'ॠ'
}

VOWELS_DEPENDENT = {
    'a': '', 'aa': 'ा', 'i': 'ि', 'ii': 'ी', 'u': 'ु', 'uu': 'ू',
    'e': 'े', 'ai': 'ै', 'o': 'ो', 'au': 'ौ',
    'ri': 'ृ'
}

SPECIALS = {
    'm̐': 'ँ',   # chandrabindu
    'ṁ': 'ं',   # anusvara
    'ḥ': 'ः',   # visarga
    '.': '।',    # danda
    '..': '॥'
}

def transliterate_en_to_hi(text):
    text = text.lower().strip()
    output = []
    i = 0
    prev_was_consonant = False

    while i < len(text):
        # Try longest matches first (3,2,1)
        matched = False
        for size in [3, 2, 1]:
            chunk = text[i:i+size]
            
            # Special symbols (ṃ, ḥ, etc.)
            if chunk in SPECIALS:
                output.append(SPECIALS[chunk])
                i += size
                matched = True
                break

            # Check for consonant clusters
            if chunk in BASE_CONSONANTS:
                output.append(BASE_CONSONANTS[chunk])
                prev_was_consonant = True
                i += size
                matched = True
                break

            # Vowel handling
            if chunk in VOWELS_INDEPENDENT:
                if prev_was_consonant:
                    # attach matra
                    output[-1] += VOWELS_DEPENDENT[chunk]
                else:
                    # independent vowel
                    output.append(VOWELS_INDEPENDENT[chunk])
                prev_was_consonant = False
                i += size
                matched = True
                break

        if not matched:
            # If nothing matched, just append raw char
            output.append(text[i])
            prev_was_consonant = False
            i += 1

    return ''.join(output)

def load_data(data_path):
    """Load English-Hindi transliteration pairs"""
    with open(data_path, "r") as f:
        data = []
        for line in f:
            line_dict = json.loads(line)
            english = line_dict["english word"]
            hindi = line_dict["native word"]
            data.append((english, hindi))

    return data


def build_vocab(data_pairs, min_freq=1):
    """Build vocabulary for source and target languages"""
    src_chars = Counter()
    tgt_chars = Counter()

    for src_word, tgt_word in data_pairs:
        src_chars.update(src_word.lower())
        tgt_chars.update(tgt_word)

    # Add special tokens
    special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    src_vocab = special_tokens + [
        char for char, freq in src_chars.items() if freq >= min_freq
    ]
    tgt_vocab = special_tokens + [
        char for char, freq in tgt_chars.items() if freq >= min_freq
    ]

    return src_vocab, tgt_vocab


def compute_accuracy(actuals, predictions):
    correct = sum(a == p for a, p in zip(actuals, predictions))
    total = len(actuals)
    accuracy = correct / total if total > 0 else 0
    return accuracy


def get_LCS(actual, prediction):
    edit_dist = lev.distance(actual, prediction, weights=(1, 1, 1e5))
    lcs_length = (len(actual) + len(prediction) - edit_dist) // 2
    return lcs_length


def get_f1_score(actuals, predictions):
    f1_scores = []
    for actual, pred in zip(actuals, predictions):
        lcs_length = get_LCS(actual, pred)
        recall = lcs_length / len(actual) if len(actual) > 0 else 0
        precision = lcs_length / len(pred) if len(pred) > 0 else 0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return average_f1
