from utils import *
import pickle as pkl

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

test_pairs = load_data("../hin/hin_test.jsonl")

preds = []
actuals = []

for en, hi in test_pairs:
    pred_hi = transliterate_en_to_hi(en)
    preds.append(pred_hi)
    actuals.append(hi)
    # print(f"EN: {en}  -->  HI: {pred_hi} (Expected: {hi})")
    # print("-" * 50)

acc = compute_accuracy(actuals, preds)
f1 = get_f1_score(actuals, preds)

with open("rule_based_results.txt", "w") as f:
    print(f"Rule-Based Transliteration Accuracy: {acc*100:.2f}%", file=f)
    print(f"Rule-Based Transliteration F1 Score: {f1*100:.2f}%", file=f)
