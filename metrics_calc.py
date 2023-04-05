import textstat

from lexical_diversity import lex_div as ld
from nltk.tokenize import word_tokenize
from torchmetrics.text.rouge import ROUGEScore


original = ""
generate = ""


def calc_lexical_diversity(text):
    t = word_tokenize(text)

    diversity_score = ld.mtld(t)
    return round(diversity_score, 2)


def calc_readability(text):
    textstat.set_lang("de")

    flesch_index = textstat.flesch_reading_ease(text)
    return round(flesch_index, 2)


def calc_rouge_score(original, generate):
    rouge = ROUGEScore(user_stemmer=True, accumulate="avg", rouge_keys=("rouge1"))

    rouge_score = rouge(generate, original)
    rouge_score = rouge_score["rouge1_fmeasure"].item()
    return round(rouge_score, 2)


print(f"\nLexical Diversity: {calc_lexical_diversity(generate)}")
print(f"\nReadability: {calc_readability(generate)}")
print(f"\nRouge Score: {calc_rouge_score(original, generate)}\n")