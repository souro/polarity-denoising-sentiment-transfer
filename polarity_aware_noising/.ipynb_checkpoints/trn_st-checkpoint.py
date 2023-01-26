import os
import dill

import torch
import torch.nn as nn

from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
from spacy.symbols import ORTH
import numpy as np

import random
import math
import time
#import sentiment

import json

#import SentimentTransfer_Evaluations as snt_ev

######################################################################################################################
config_file = open("config.json")
config_obj = json.load(config_file)

json_dumps = json.dumps(config_obj)
print(json_dumps, flush=True)

dir = config_obj["dir"]
field_fix_length = config_obj["field_fix_length"]

# train_data_path = config_obj["train_data_path"]
# valid_data_path = config_obj["valid_data_path"]
# test_data_path = config_obj["test_data_path"]

train_pos_data_path = config_obj["train_pos_data_path"]
train_neg_data_path = config_obj["train_neg_data_path"]
valid_pos_data_path = config_obj["valid_pos_data_path"]
valid_neg_data_path = config_obj["valid_neg_data_path"]
test_pos_data_path = config_obj["test_pos_data_path"]
test_neg_data_path = config_obj["test_neg_data_path"]

src_ext = config_obj["src_ext"]
trg_ext = config_obj["trg_ext"]
vocab_min_freq = config_obj["vocab_min_freq"]
vocab_max_size = config_obj["vocab_max_size"]
batch_size = config_obj["batch_size"]
encoder_max_length = config_obj["encoder_max_length"]
decoder_max_length = config_obj["decoder_max_length"]
style_num_embeddings = config_obj["style_num_embeddings"]
style_embedding_dim = config_obj["style_embedding_dim"]
hid_dim_enc = config_obj["hid_dim_enc"]
hid_dim_dec = config_obj["hid_dim_dec"]
enc_layers = config_obj["enc_layers"]
dec_layers = config_obj["dec_layers"]
enc_heads = config_obj["enc_heads"]
dec_heads = config_obj["dec_heads"]
enc_pf_dim = config_obj["enc_pf_dim"]
dec_pf_dim = config_obj["dec_pf_dim"]
enc_dropout = config_obj["enc_dropout"]
dec_dropout = config_obj["dec_dropout"]
learning_rate = config_obj["learning_rate"]
add_loss_t_s_start_epoch = config_obj["add_loss_t_s_start_epoch"]
add_loss_t_s_end_epoch = config_obj["add_loss_t_s_end_epoch"]
add_loss_t_w_start_epoch = config_obj["add_loss_t_w_start_epoch"]
add_loss_t_w_end_epoch = config_obj["add_loss_t_w_end_epoch"]
add_loss_s_w_start_epoch = config_obj["add_loss_s_w_start_epoch"]
add_loss_s_w_end_epoch = config_obj["add_loss_s_w_end_epoch"]
add_loss_t_s_w_start_epoch = config_obj["add_loss_t_s_w_start_epoch"]
add_loss_t_s_w_end_epoch = config_obj["add_loss_t_s_w_end_epoch"]
alt_loss_t_s_start_epoch = config_obj["alt_loss_t_s_start_epoch"]
alt_loss_t_s_end_epoch = config_obj["alt_loss_t_s_end_epoch"]
alt_loss_t_w_start_epoch = config_obj["alt_loss_t_w_start_epoch"]
alt_loss_t_w_end_epoch = config_obj["alt_loss_t_w_end_epoch"]
alt_loss_s_w_start_epoch = config_obj["alt_loss_s_w_start_epoch"]
alt_loss_s_w_end_epoch = config_obj["alt_loss_s_w_end_epoch"]
alt_loss_t_s_w_start_epoch = config_obj["alt_loss_t_s_w_start_epoch"]
alt_loss_t_s_w_end_epoch = config_obj["alt_loss_t_s_w_end_epoch"]
translation_loss_start_epoch = config_obj["translation_loss_start_epoch"]
translation_loss_end_epoch = config_obj["translation_loss_end_epoch"]
style_loss_start_epoch = config_obj["style_loss_start_epoch"]
style_loss_end_epoch = config_obj["style_loss_end_epoch"]
words_style_loss_start_epoch = config_obj["words_style_loss_start_epoch"]
words_style_loss_end_epoch = config_obj["words_style_loss_end_epoch"]

num_epochs = config_obj["num_epochs"]
clip = config_obj["clip"]
early_stop_lookout = config_obj["early_stop_lookout"]
another_early_stop_lookout = config_obj["another_early_stop_lookout"]
add_loss_t_s = config_obj["add_loss_t_s"]
add_loss_t_w = config_obj["add_loss_t_w"]
add_loss_s_w = config_obj["add_loss_s_w"]
add_loss_t_s_w = config_obj["add_loss_t_s_w"]
alt_loss_t_s = config_obj["alt_loss_t_s"]
alt_loss_t_w = config_obj["alt_loss_t_w"]
alt_loss_s_w = config_obj["alt_loss_s_w"]
alt_loss_t_s_w = config_obj["alt_loss_t_s_w"]
t_loss = config_obj["translation_loss"]
s_s_loss = config_obj["style_loss"]
w_s_loss = config_obj["words_style_loss"]
v_p_loss = config_obj["vocab_prob_loss"]
v_p_loss_start_epoch = config_obj["vocab_prob_loss_start_epoch"]
v_p_loss_end_epoch = config_obj["vocab_prob_loss_end_epoch"]
add_loss_t_v = config_obj["add_loss_t_v"]
add_loss_t_v_start_epoch = config_obj["add_loss_t_v_start_epoch"]
add_loss_t_v_end_epoch = config_obj["add_loss_t_v_end_epoch"]
alt_loss_t_v = config_obj["alt_loss_t_v"]
alt_loss_t_v_start_epoch = config_obj["alt_loss_t_v_start_epoch"]
alt_loss_t_v_end_epoch = config_obj["alt_loss_t_v_end_epoch"]
translation_loss_weight = config_obj["translation_loss_weight"]
vocab_loss_weight = config_obj["vocab_loss_weight"]
ss_loss_weight = config_obj["ss_loss_weight"]
ws_loss_weight = config_obj["ws_loss_weight"]
check_best_after_epoch = config_obj["check_best_after_epoch"]
check_best_after_epoch2 = config_obj["check_best_after_epoch2"]
specific_epoch_checkpoint = config_obj["specific_epoch_checkpoint"]
debug = config_obj["debug"]
style_cond = config_obj["style_cond"]
is_only_evaluation = config_obj["is_only_evaluation"]
pretrn_encoder = config_obj["pretrn_encoder"]
#######################################################################################################################
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
#######################################################################################################################

spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('de_core_news_sm')
spacy_fr.tokenizer.add_special_case(u'<style>', [{ORTH: u'<style>'}])

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = batch_size

#######################################################################################################################
def load_field(path):
    with open(path, 'rb') as f:
        return dill.load(f)

#Please provide the path for the pretrained src field from the noisy translation
SRC = load_field(os.path.join("", 'src.field'))

TRG = Field(tokenize = tokenize_en,
            # tokenize = 'spacy',
            # tokenizer_language='de',
            init_token='<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True,
            fix_length=field_fix_length)

train_pos_data = TranslationDataset(
    path=train_neg_data_path+str(0),
    exts=(src_ext, trg_ext),
    fields=(SRC, TRG),
)

TRG.build_vocab(train_pos_data, min_freq=vocab_min_freq, max_size=vocab_max_size)
#######################################################################################################################
print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}", flush=True)
print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}", flush=True)
#######################################################################################################################

#train_data, valid_data, test_data = WMT14.splits(exts=('.en', '.de'), fields=(SRC, TRG))

# train_data = TranslationDataset(
#     path=train_data_path,
#     exts=(src_ext, trg_ext),
#     fields=(SRC, TRG),
# )
# valid_data = TranslationDataset(
#     path=valid_data_path,
#     exts=(src_ext, trg_ext),
#     fields=(SRC, TRG),
# )
# test_data = TranslationDataset(
#     path=test_data_path,
#     exts=(src_ext, trg_ext),
#     fields=(SRC, TRG),
# )
test_pos_data = TranslationDataset(
    path=test_neg_data_path+str(0),
    exts=(src_ext, trg_ext),
    fields=(SRC, TRG),
)
test_neg_data = TranslationDataset(
    path=test_pos_data_path+str(0),
    exts=(src_ext, trg_ext),
    fields=(SRC, TRG),
)

#######################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = batch_size

def data_read(epoch):
    train_pos_data = TranslationDataset(
        path=train_neg_data_path+str(epoch),
        exts=(src_ext, trg_ext),
        fields=(SRC, TRG),
    )
    # train_neg_data = TranslationDataset(
    #     path=train_neg_data_path,
    #     exts=(src_ext, trg_ext),
    #     fields=(SRC, TRG),
    # )
    valid_pos_data = TranslationDataset(
        path=valid_neg_data_path+str(epoch),
        exts=(src_ext, trg_ext),
        fields=(SRC, TRG),
    )
    # valid_neg_data = TranslationDataset(
    #     path=valid_neg_data_path,
    #     exts=(src_ext, trg_ext),
    #     fields=(SRC, TRG),
    # )
    test_pos_data = TranslationDataset(
        path=test_neg_data_path+str(epoch),
        exts=(src_ext, trg_ext),
        fields=(SRC, TRG),
    )
    # test_neg_data = TranslationDataset(
    #     path=test_neg_data_path,
    #     exts=(src_ext, trg_ext),
    #     fields=(SRC, TRG),
    # )
    #######################################################################################################################
    print(f"Number of training examples: {len(train_pos_data.examples)}", flush=True)
    #print(f"Number of training examples: {len(train_neg_data.examples)}", flush=True)

    print(f"Number of validation examples: {len(valid_pos_data.examples)}", flush=True)
    #print(f"Number of validation examples: {len(valid_neg_data.examples)}", flush=True)

    print(f"Number of testing examples: {len(test_pos_data.examples)}", flush=True)

    # print(f"Number of training examples: {len(train_data.examples)}", flush=True)
    # print(f"Number of validation examples: {len(valid_data.examples)}", flush=True)
    # print(f"Number of testing examples: {len(test_data.examples)}", flush=True)
    #######################################################################################################################
    # train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    #     (train_data, valid_data, test_data),
    train_pos_iterator, valid_pos_iterator, test_pos_iterator = BucketIterator.splits(
        (train_pos_data, valid_pos_data, test_pos_data),
        batch_size = BATCH_SIZE,
        sort_within_batch=True,
        sort_key= lambda x: len(x.src),
        device = device)
    return train_pos_iterator, valid_pos_iterator, test_pos_iterator
######################################################################################################################
class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=encoder_max_length):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src
#######################################################################################################################
class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src
#######################################################################################################################
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention
#######################################################################################################################
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x
#######################################################################################################################
class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=decoder_max_length):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention
#######################################################################################################################
class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention
#######################################################################################################################
class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)


        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention

class Seq2Seq2(nn.Module):
    def __init__(self,
                 pretrnd_model,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.pretrnd_model = pretrnd_model
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.pretrnd_model.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)


        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention
######################################################################################################################
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
#HID_DIM = 512
HID_DIM_ENC = hid_dim_enc
HID_DIM_DEC = hid_dim_dec
ENC_LAYERS = enc_layers
DEC_LAYERS = dec_layers
ENC_HEADS = enc_heads
DEC_HEADS = dec_heads
ENC_PF_DIM = enc_pf_dim
DEC_PF_DIM = dec_pf_dim
ENC_DROPOUT = enc_dropout
DEC_DROPOUT = dec_dropout

enc = Encoder(INPUT_DIM,
              HID_DIM_ENC,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)
dec_pretrained = Decoder(15505,
              HID_DIM_DEC,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM_DEC,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)
######################################################################################################################
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

pretrnd_model = Seq2Seq(enc, dec_pretrained, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
wmt_checkpoint = torch.load("../../simple_translation/"+pretrn_encoder+"/checkpoint.pt")
pretrnd_model.load_state_dict(wmt_checkpoint['state_dict'])
model = Seq2Seq2(pretrnd_model, enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

#######################################################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters', flush=True)
#######################################################################################################################
# def initialize_weights(m):
#     if hasattr(m, 'weight') and m.weight.dim() > 1:
#         nn.init.xavier_uniform_(m.weight.data)
# if is_only_evaluation == False:
#     model.apply(initialize_weights)
#######################################################################################################################
# pretrained_dict = torch.load("../../simple_translation/wmt_translation"+'/checkpoint.pt')['state_dict']
# # with open('dict_keys', 'w')as op:
# # for k, v in torch.load("../../simple_translation/wmt_translation"+'/checkpoint.pt')['state_dict'].items():
# #     op.write(k)
# #     op.write('\n')
# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
# enc_dict = {}
# enc_keys = ['encoder.'+k for k in model.encoder.state_dict().keys()]
# for k, v in pretrained_dict.items():
#     if k in enc_keys:
#         enc_dict[k.replace('encoder.', '')]=v
# model.encoder.state_dict().update(enc_dict)
# model.encoder.load_state_dict(enc_dict)
#######################################################################################################################
LEARNING_RATE = learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
#######################################################################################################################
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
#######################################################################################################################
# checkpoint = torch.load(dir+'/checkpoint_7.pt')
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# save_epoch = checkpoint['epoch']
#######################################################################################################################
def loss_translation(criterion, output, trg):
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    trg = trg[:, 1:].contiguous().view(-1)
    # output = [batch size * trg len - 1, output dim]
    # trg = [batch size * trg len - 1]
    translation_loss = criterion(output, trg)
    return translation_loss

#######################################################################################################################
pos_list_enc = [3 for i in range(100)]
neg_list_enc = [4 for i in range(100)]

pos_list_dec = [3 for i in range(99)]
neg_list_dec = [4 for i in range(99)]

def calculate_output_loss(epoch, model, pos_iterator, criterion, is_eval, optimizer=None, clip=None):
    epoch_translation_loss = 0

    total_no_batches = 0

    for i, batch in enumerate(pos_iterator):
        src = batch.src
        trg = batch.trg

        if is_eval==False:
            optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        translation_loss = loss_translation(criterion, output, trg)

        if is_eval == False:
            translation_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        epoch_translation_loss += translation_loss.item()
        total_no_batches +=1

    epoch_translation_loss_avg = epoch_translation_loss / total_no_batches

    return epoch_translation_loss_avg

def train(epoch, model, pos_iterator, criterion, is_eval, optimizer=None, clip=None):
    if is_eval == False:
        model.train()
        epoch_translation_loss = calculate_output_loss(epoch, model, pos_iterator, criterion, False, optimizer, clip)
    else:
        model.eval()
        with torch.no_grad():
            epoch_translation_loss = calculate_output_loss(epoch, model, pos_iterator, criterion, True)
    return epoch_translation_loss
#######################################################################################################################
import functools
def post_processing(text_list):
    repl_list = {'@@ ': '', '<eos>':''}
    text_str = ' '.join(text_list)
    return functools.reduce(lambda a, kv: a.replace(*kv), repl_list.items(), text_str)
############################################################################################################################
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
#######################################################################################################################
if is_only_evaluation == False:
    N_EPOCHS = num_epochs
    CLIP = clip

    best_valid_loss = float('inf')
    valid_loss_before = float('inf')

    early_stop_cnt = 0
    early_stop_lookout = early_stop_lookout
    early_stop=False

    another_early_stop_cnt = 0
    another_early_stop_lookout = another_early_stop_lookout
    another_early_stop=False

    best_epoch_no = 0

    #for epoch in range(save_epoch, N_EPOCHS):
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_pos_iterator, valid_pos_iterator, test_pos_iterator = data_read(epoch)
        train_translation_loss = train(epoch, model, train_pos_iterator, criterion, False, optimizer, CLIP)
        valid_translation_loss = train(epoch, model, valid_pos_iterator, criterion, True)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, dir + f'/latest_checkpoint.pt')
        if(epoch == specific_epoch_checkpoint):
            torch.save(checkpoint, dir + f'/checkpoint_{specific_epoch_checkpoint}.pt')

        if(epoch > check_best_after_epoch):
            #Early_Stop
            if valid_translation_loss < best_valid_loss or valid_translation_loss < valid_loss_before:
                early_stop_cnt = 0
                early_stop = False

            elif valid_translation_loss >= best_valid_loss or valid_translation_loss >= valid_loss_before:
                early_stop_cnt += 1
                early_stop = True

            # Another Early_Stop based on only best valid translation_loss
            if epoch > check_best_after_epoch2 :
                if valid_translation_loss < best_valid_loss:
                    another_early_stop_cnt = 0
                    another_early_stop = False

                elif valid_translation_loss >= best_valid_loss:
                    another_early_stop_cnt += 1
                    another_early_stop = True


            if valid_translation_loss < best_valid_loss:
                best_valid_loss = valid_translation_loss
                best_epoch_no = epoch

                torch.save(checkpoint, dir + '/best_valid_checkpoint.pt')
                # if(epoch==4):
                #     torch.save(checkpoint, dir+'/epoch5_checkpoint.pt')

            valid_loss_before = valid_translation_loss


        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s', flush=True)
        print(f'\tTrain Translation Loss: {train_translation_loss:.7f}', flush=True)
        print(f'\t Val. Translation Loss: {valid_translation_loss:.3f}', flush=True)
        print(f'\t Till now Best Val. Loss: {best_valid_loss:.3f} found on {best_epoch_no+1} epoch ', flush=True)

        if early_stop==True :
            print(f'EarlyStopping counter (1st way): {early_stop_cnt} out of {early_stop_lookout}', flush=True)

        if another_early_stop==True :
            print(f'EarlyStopping counter (2nd way): {another_early_stop_cnt} out of {another_early_stop_lookout}', flush=True)

        print('\n', flush=True)


        if early_stop_cnt == early_stop_lookout:
            print('Early Stoping (1st way)...', flush=True)
            break
        if another_early_stop_cnt == another_early_stop_lookout:
            print('Early Stoping (2nd way)...', flush=True)
            break
#######################################################################################################################
latest_checkpoint = torch.load(dir+'/latest_checkpoint.pt')
best_valid_checkpoint = torch.load(dir+'/best_valid_checkpoint.pt')

model.load_state_dict(best_valid_checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#epoch = checkpoint['epoch']

if is_only_evaluation == False:
    test_loss = train(None, model, test_pos_iterator, criterion, True)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |', flush=True)
#######################################################################################################################
# def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
#     assert n_rows * n_cols == n_heads
#
#     fig = plt.figure(figsize=(15, 25))
#
#     for i in range(n_heads):
#         ax = fig.add_subplot(n_rows, n_cols, i + 1)
#
#         _attention = attention.squeeze(0)[i].cpu().detach().numpy()
#
#         cax = ax.matshow(_attention, cmap='bone')
#
#         ax.tick_params(labelsize=12)
#         ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
#                            rotation=45)
#         ax.set_yticklabels([''] + translation)
#
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#     plt.close()
#######################################################################################################################
def translate_sentence(sentence, src_field, trg_field, model, device, max_len, decoder_flag, is_st):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    if(is_st == True):
        if(decoder_flag=='pos'):
            decoder_flag = 'neg'
        elif(decoder_flag=='neg'):
            decoder_flag = 'pos'

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[tokens[0]]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            # if(decoder_flag == 'pos'):
            #     output, attention = model.pos_decoder(trg_tensor, enc_src, trg_mask, src_mask)
            # elif(decoder_flag == 'neg'):
            #     output, attention = model.neg_decoder(trg_tensor, enc_src, trg_mask, src_mask)
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention
#######################################################################################################################
#display_attention(src, translation, attention)
######################################################################################################################
from torchtext.data.metrics import bleu_score
import nltk
senti_trg_list_ops = []
def masked_sent(sent_list):
    masked = None #snt_ev.mask_polarity(post_processing(sent_list))
    return nltk.word_tokenize(masked)

def calculate_bleu(data, src_field, trg_field, model, device, max_len, decoder_flag, is_st):
    trgs = []
    masked_trgs = []

    pred_trgs = []
    masked_pred_trgs = []

    lengthy_idx = []
    for idx, datum in enumerate(data):
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        if len(src) < 100 and len(trg) < 100:
            pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len, decoder_flag, is_st)

            # cut off <eos> token
            pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            #masked_pred_trgs.append(masked_sent(pred_trg))
            trgs.append([trg])
            #masked_trgs.append([masked_sent(trg)])
        else:
            lengthy_idx.append(idx)

    # print(lengthy_idx)
    # print(len(pred_trgs))
    return bleu_score(pred_trgs, trgs), None #bleu_score(masked_pred_trgs, masked_trgs)
# ######################################################################################################################
# b_score, masked_b_score = calculate_bleu(test_pos_data, SRC, TRG, model, device, 50, 'pos', True)
# print('\n')
# print(f'Style Transfer: BLEU score on test pos data = {b_score*100:.5f}', flush=True)
# print(f'Style Transfer: Masked BLEU score on test pos data = {masked_b_score*100:.5f}', flush=True)
b_score, masked_b_score = calculate_bleu(test_neg_data, SRC, TRG, model, device, 50, 'neg', True)
print('\n')
print(f'Style Transfer: BLEU score on test neg data = {b_score*100:.5f}', flush=True)
#print(f'Style Transfer: Masked BLEU score on test neg data = {masked_b_score*100:.5f}', flush=True)
print('\n')
#####################################################################################################################
def evaluation(data, decoder_flag):
    correct_count = 0
    lm_scores = []
    similarity_scores = []
    masked_similarity_scores = []

    for idx in range(1000):
        example_idx = idx

        src = vars(data.examples[example_idx])['src']
        trg = vars(data.examples[example_idx])['trg']


        #predicted_trg_trn, attention = translate_sentence(src, SRC, TRG, model, device, 50, decoder_flag, False)
        predicted_trg_st, attention = translate_sentence(src, SRC, TRG, model, device, 50, decoder_flag, True)

        print(f'src = {post_processing(src)}', flush=True)

        print(f'trg = {post_processing(trg)}', flush=True)
        result_trg = snt_ev.senti_score(post_processing(trg))
        print("Label:", result_trg['label'])
        print("Confidence Score:", result_trg['score'])


        print(f'style transfered predicted trg = {post_processing(predicted_trg_st)}', flush=True)

        # Sentiment Score
        result_pred = snt_ev.senti_score(post_processing(predicted_trg_st))
        print("Label:", result_pred['label'])
        print("Confidence Score:", result_pred['score'])

        if (result_trg['label'] != result_pred['label']):
            correct_count += 1

        #LM Score
        gpt_lm_score = snt_ev.lm_score(post_processing(predicted_trg_st))
        print("LM Score:", gpt_lm_score)
        lm_scores.append(gpt_lm_score)

        #Similarity Score
        similarity_score = snt_ev.similarity(post_processing(trg), post_processing(predicted_trg_st))
        print('Similarity Score: ', similarity_score)
        similarity_scores.append(similarity_score)
        similarity_score_masked = snt_ev.similarity(snt_ev.mask_polarity(post_processing(trg)), snt_ev.mask_polarity(post_processing(predicted_trg_st)))
        print('Masked Similarity Score: ', similarity_score_masked)
        masked_similarity_scores.append(similarity_score_masked)
        ###

        print('\n', flush=True)

    lm_scores_mean = sum(lm_scores) / len(lm_scores)
    similarity_scores_mean = sum(similarity_scores) / len(similarity_scores)
    masked_similarity_scores_mean = sum(masked_similarity_scores) / len(masked_similarity_scores)


    return correct_count, lm_scores_mean, similarity_scores_mean, masked_similarity_scores_mean



def imdb_op(data, decoder_flag):
    correct_count = 0
    lm_scores = []
    similarity_scores = []
    masked_similarity_scores = []
    with open('imdb_outputours2.txt', 'w') as imdb_f:
        for idx in range(1000):
            example_idx = idx

            src = vars(data.examples[example_idx])['src']
            trg = vars(data.examples[example_idx])['trg']


        
            predicted_trg_st, attention = translate_sentence(src, SRC, TRG, model, device, 50, decoder_flag, True)

            imdb_f.write(post_processing(predicted_trg_st)+'\n')
imdb_op(test_neg_data, 'neg')
########################################################################################################################
# model.load_state_dict(best_valid_checkpoint['state_dict'])
# print("Training Positive data", flush=True)
# print('##############################################################################################################')
# evaluation(train_pos_data, 'pos')
# print('##############################################################################################################')
# print("Training Negative data", flush=True)
# print('##############################################################################################################')
# evaluation(train_neg_data, 'neg')
##################################################################################################################
#model.load_state_dict(best_valid_checkpoint['state_dict'])
# print("Testing Positive data", flush=True)
# print('##############################################################################################################')
# #evaluation(test_pos_data, 'pos')
# correct_count, lm_scores_mean, similarity_scores_mean, masked_similarity_scores_mean = evaluation(test_pos_data, 'pos')
# print(correct_count)
# print(lm_scores_mean)
# print(similarity_scores_mean)
# print(masked_similarity_scores_mean)
print('##############################################################################################################')
print("Testing Negative data", flush=True)
print('##############################################################################################################')
#evaluation(test_neg_data, 'neg')
#correct_count, lm_scores_mean, similarity_scores_mean, masked_similarity_scores_mean = evaluation(test_neg_data, 'neg')
#print(correct_count)
#print(lm_scores_mean)
#print(correct_count)
#print(lm_scores_mean)
#print(similarity_scores_mean)
#print(masked_similarity_scores_mean)
