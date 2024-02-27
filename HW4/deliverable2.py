from typing import List
import numpy as np
import math
from collections import Counter

def bleu_score(predicted: List[int], target: List[int], N: int) -> float:
    """
    Finds the BLEU-N score between the predicted sentence and a single reference (target) sentence.
    Feel free to deviate from this skeleton if you prefer.

    Edge case: If the length of the predicted sentence or the target is less than N, return 0.
    """
    if len(predicted) < N or len(target) < N:
        # TODO
        return 0.0
    
    def C(y, g):
        # TODO how many times does n-gram g appear in y?
        y_counter = Counter([tuple(y[i:i+len(g)]) for i in range(len(y)-len(g)+1)])
        return y_counter[g]
    
    def clipped_precision(predicted, target, n):
        predicted_ngrams = [tuple(predicted[i:i+n]) for i in range(len(predicted)-n+1)]
        target_ngrams = [tuple(target[i:i+n]) for i in range(len(target)-n+1)]
        
        predicted_ngram_counts = Counter(predicted_ngrams)
        target_ngram_counts = Counter(target_ngrams)

        clipped_counts = {ngram: min(count, target_ngram_counts[ngram]) for ngram, count in predicted_ngram_counts.items()}

        numerator = sum(clipped_counts.values())
        denominator = len(predicted) - n + 1

        return numerator, denominator

    geo_mean = 1.0
    for n in range(1, N+1):
        numerator, denominator = clipped_precision(predicted, target, n)
        if denominator == 0:
            return 0.0
        geo_mean *= (numerator/denominator)
    geo_mean = geo_mean **(1/N)
    
    brevity_penalty =  min(1,  math.exp(1 - len(target) / len(predicted))) # TODO
    return brevity_penalty * geo_mean


