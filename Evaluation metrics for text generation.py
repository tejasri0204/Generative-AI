pip install nltk rouge-score

import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Reference (ground truth) text
reference_text = "Once upon a time, there was a curious cat and a clever mouse. They lived in a cozy little house in the countryside."

# Generated text
generated_text = "In a small village, there lived a curious cat and a clever mouse. They shared a cozy house."

# BLEU Score
reference = [reference_text.split()]
candidate = generated_text.split()
bleu_score = sentence_bleu(reference, candidate)

 ROUGE Score
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = scorer.score(reference_text, generated_text)
rouge_score = scores['rougeL'].fmeasure

print("BLEU Score:", bleu_score)
print("ROUGE Score:", rouge_score)