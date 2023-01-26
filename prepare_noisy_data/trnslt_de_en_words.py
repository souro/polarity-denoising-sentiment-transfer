import argparse
import random
import pandas as pd
import torch

from torchtext.datasets import TranslationDataset
from torchtext.data import Field

import spacy
import numpy as np

import goslate

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti_analyzer = SentimentIntensityAnalyzer()

parser.add_argument('--field_fix_length', default=100)
parser.add_argument('--data_path', default='')
parser.add_argument('--src_ext', default='.de')
parser.add_argument('--trg_ext', default='.en')

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

args = parser.parse_args()

SRC = Field(tokenize=tokenize_de,
            # tokenize = 'spacy',
            # tokenizer_language='en',
            # init_token='<sos',
            # eos_token = '<eos>',
            lower=True,
            batch_first=True,
            fix_length=args.field_fix_length)

TRG = Field(tokenize=tokenize_en,
            # tokenize = 'spacy',
            # tokenizer_language='de',
            # init_token='<sos>',
            # eos_token = '<eos>',
            lower=True,
            batch_first=True,
            fix_length=args.field_fix_length)

data = TranslationDataset(
    path=args.data_path,
    exts=(args.src_ext, args.trg_ext),
    fields=(SRC, TRG),
)

print(f"Number of examples: {len(data.examples)}", flush=True)
SRC.build_vocab(data)
TRG.build_vocab(data)
print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}", flush=True)
print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}", flush=True)

def is_polarized(en_token):
    score = senti_analyzer.polarity_scores(en_token)
    res = None
    if (score['neu'] == 1.0):
        res = False, 'Neu'
    elif (score['pos'] == 1.0):
        res = True, 'Pos'
    elif (score['neg'] == 1.0):
        res = True, 'Neg'
    else:
        res = False, 'Neu'
    return res

#gs = goslate.Goslate()
#print(gs.translate('gemacht', 'en'))

def de_to_en(token):
    import requests
    url = "https://lindat.mff.cuni.cz/services/translation/api/v2/languages/?src=de&tgt=en"
    ip_text = {'input_text': token}

    response = requests.post(url, ip_text)

    return response.text.replace('\n','')

src_list = []
dest_list = []
is_polrz_list = []
plrty_list = []

# import concurrent.futures
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=200)
# gs = goslate.Goslate(executor=executor)
# it = gs.translate(SRC.vocab.itos, 'en')


for token in SRC.vocab.itos:
    src_list.append(token)
    dest = de_to_en(token)
    dest_list.append(dest)
    is_polarize, polarity = is_polarized(token)
    is_polrz_list.append(is_polarize)
    plrty_list.append(polarity)
trn_plrty_src = pd.DataFrame(
    {'german_tokens': src_list,
     'translated_english_tokens': dest_list,
     'is_polarize': is_polrz_list,
     'polarity': plrty_list
     })
trn_plrty_src.to_csv('trns_polr_src.csv')

dest_list = []
is_polrz_list = []
plrty_list = []

for token in TRG.vocab.itos:
    dest_list.append(token)
    is_polarize, polarity = is_polarized(token)
    is_polrz_list.append(is_polarize)
    plrty_list.append(polarity)
trn_plrty_trg = pd.DataFrame(
    {'english_tokens': dest_list,
     'is_polarize': is_polrz_list,
     'polarity': plrty_list
     })
trn_plrty_trg.to_csv('trns_polr_trg.csv')


