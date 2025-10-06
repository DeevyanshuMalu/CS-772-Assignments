import Levenshtein as lev

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
