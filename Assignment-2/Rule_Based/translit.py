from utils import *
import pickle as pkl
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
