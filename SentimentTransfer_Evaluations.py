from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti_analyzer = SentimentIntensityAnalyzer()

from nltk.translate.bleu_score import sentence_bleu

model = SentenceTransformer('bert-base-nli-mean-tokens')

from lm_scorer.models.auto import AutoLMScorer
scorer = AutoLMScorer.from_pretrained("gpt2-large")

from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis")

def similarity(sentence1, sentence2):
    sentence_embedding1 = model.encode(sentence1)
    sentence_embedding2 = model.encode(sentence2)
    sim_score = cosine_similarity([sentence_embedding1], [sentence_embedding2])
    return sim_score[0][0]

def mask_polarity(sentence):
    tokens = nltk.word_tokenize(sentence)
    sent_toks = []
    continuous = False
    for token in tokens:
        score = senti_analyzer.polarity_scores(token)
        # print(token)
        # print(score)
        # print("\n")

        if (score['pos'] == 1.0) or (score['neg'] == 1.0) :
            if continuous == False:
                sent_toks.append("<setiment>")
                continuous = True
        else:
            sent_toks.append(token)
            continuous = False
    return " ".join(sent_toks)

def senti_score(sentence):
    return sentiment_analysis(sentence)[0]

def lm_score(sentence):
    gpt_lm_score = scorer.sentence_score(sentence, log=True)
    return gpt_lm_score

def main():
    sentence1 = "i was very disappointed and had forgotten it before it was even finished good bad this good bad great."
    print(sentence1)
    sentence2 = "i was very happy and had remember it when it was complete good this bad great."
    print(sentence2)
    #Similarity
    print(similarity(sentence1, sentence2))
    #Bleu Score
    print (sentence_bleu([sentence1.split()], sentence2.split())*100)
    #Lm Score
    print(lm_score(sentence1))
    print(lm_score(sentence2))

    sentence1_msak = mask_polarity(sentence1)
    print(sentence1_msak)
    sentence2_msak = mask_polarity(sentence2)
    print(sentence2_msak)
    #Similarity
    print(similarity(sentence1_msak, sentence2_msak))
    #Bleu Score
    print (sentence_bleu([sentence1_msak.split()], sentence2_msak.split())*100)

if __name__=="__main__":
    main()
