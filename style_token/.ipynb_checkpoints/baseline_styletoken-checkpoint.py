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
import sentiment

import ../SentimentTransfer_Evaluations as snt_ev
#######################################################################################################################
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
#######################################################################################################################

spacy_en = spacy.load('en_core_web_sm')
spacy_en.tokenizer.add_special_case(u'<n>', [{ORTH: u'<n>'}])
spacy_en.tokenizer.add_special_case(u'<p>', [{ORTH: u'<p>'}])
spacy_fr = spacy.load('de_core_news_sm')
spacy_fr.tokenizer.add_special_case(u'<n>', [{ORTH: u'<n>'}])
spacy_fr.tokenizer.add_special_case(u'<p>', [{ORTH: u'<p>'}])

def tokenize_fr(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

#######################################################################################################################
SRC = Field(tokenize = tokenize_fr,
            # tokenize = 'spacy',
            # tokenizer_language='en',
            #init_token='<sos',
            eos_token = '<eos>',
            lower = True,
            batch_first = True,
            fix_length=100)

TRG = Field(tokenize = tokenize_en,
            # tokenize = 'spacy',
            # tokenizer_language='de',
            #init_token='<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True,
            fix_length=100)
#######################################################################################################################
# Please provide the path for the data
train_data = TranslationDataset(
    path = "",
    exts=(".de", ".en"),
    fields=(SRC, TRG),
)
valid_data = TranslationDataset(
    path = "",
    exts=(".de", ".en"),
    fields=(SRC, TRG),
)
test_data = TranslationDataset(
    path = "",
    exts=(".de", ".en"),
    fields=(SRC, TRG),
)
#######################################################################################################################
print(f"Number of training examples: {len(train_data.examples)}", flush=True)
print(f"Number of validation examples: {len(valid_data.examples)}", flush=True)
print(f"Number of testing examples: {len(test_data.examples)}", flush=True)
# print(vars(train_data.examples[:5]), flush=True)
# print(vars(valid_data.examples[:5]), flush=True)
# print(vars(test_data.examples[:5]), flush=True)
#######################################################################################################################
specials=['<p>', '<n>']

SRC.build_vocab(train_data, min_freq = 2, max_size=30000, specials=specials)

TRG.build_vocab(train_data, min_freq = 2, max_size=30000, specials=specials)
#######################################################################################################################
print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}", flush=True)
print(f"Unique tokens in target (fr) vocabulary: {len(TRG.vocab)}", flush=True)
#######################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################################################################################
BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch=True,
    sort_key= lambda x: len(x.src),
    device = device)
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
                 max_length=100):
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

    def forward(self, src, style_embedd_enc, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        #src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos) + style_embedd_enc)
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
                 max_length=100):
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

    def forward(self, trg, enc_src, style_embedd_dec, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        #trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos) + style_embedd_dec)
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
        self.style_embedding = nn.Embedding(5, 512)

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

    def forward(self, src, style_tok_list_tensor_enc, style_tok_list_tensor_dec, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        style_embedd_enc = self.style_embedding(style_tok_list_tensor_enc)
        style_embedd_dec = self.style_embedding(style_tok_list_tensor_dec)

        enc_src = self.encoder(src, style_embedd_enc, src_mask)

        # enc_src = [batch size, src len, hid dim]
        # enc_src1 = style_embedd.add(enc_src_before)
        # enc_src = torch.div(enc_src1,2)

        # enc_src = torch.cat((enc_src_before, style_embedd), dim=2)

        output, attention = self.decoder(trg, enc_src, style_embedd_dec, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention
######################################################################################################################
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
#HID_DIM = 512
HID_DIM_ENC = 512
HID_DIM_DEC = 512
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM_ENC,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
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

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
#######################################################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters', flush=True)
#######################################################################################################################
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
#######################################################################################################################
model.apply(initialize_weights)
#######################################################################################################################
LEARNING_RATE = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
#######################################################################################################################
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
#######################################################################################################################
# checkpoint = torch.load('baseline_styletoken_100k/checkpoint.pt')
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# save_epoch = checkpoint['epoch']
#######################################################################################################################
pos_list_enc = [3 for i in range(100)]
neg_list_enc = [4 for i in range(100)]

pos_list_dec = [3 for i in range(99)]
neg_list_dec = [4 for i in range(99)]

def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train()

    epoch_loss = 0
    epoch_style_loss = 0
    epoch_translation_loss = 0

    loss_bool = True

    list_translation_loss = []
    list_style_loss = []
    loss_counter = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        #torch.set_printoptions(edgeitems=50)
        #print(src)
        trg = batch.trg

        style_tok_list_enc = []
        style_tok_list_dec = []
        style_tok_list_dec_ops = []
        senti_trg_list = []
        senti_trg_list_ops = []

        # import pudb
        # pudb.set_trace()

        for i, x in enumerate(src.cpu().numpy()):
            # for j, k in enumerate(x):
            if (x[0] == 3):
                #senti_trg_list_ops.append(0)
                senti_trg_list_ops.append(0)
                style_tok_list_enc.append(pos_list_enc)
                style_tok_list_dec.append(pos_list_dec)
                style_tok_list_dec_ops.append(neg_list_dec)
            elif (x[0] == 4):
                senti_trg_list_ops.append(0)
                style_tok_list_enc.append(neg_list_enc)
                style_tok_list_dec.append(neg_list_dec)
                style_tok_list_dec_ops.append(pos_list_dec)
        style_tok_list_np_enc = np.array(style_tok_list_enc)
        style_tok_list_np_dec = np.array(style_tok_list_dec)
        style_tok_list_np_dec_ops = np.array(style_tok_list_dec_ops)
        style_tok_list_tensor_enc = torch.from_numpy(style_tok_list_np_enc).to(device)
        style_tok_list_tensor_dec = torch.from_numpy(style_tok_list_np_dec).to(device)
        style_tok_list_tensor_dec_ops = torch.from_numpy(style_tok_list_np_dec_ops).to(device)

        trg_copy = trg.detach().clone()
        trg_ops_list = []
        for i, x in enumerate(trg_copy.cpu().numpy()):
            # if (x[0]==3):
            #     x[0]=4
            # elif (x[0]==4):
            #     x[0]=3
            if (x[0] == 3):
                x[0] = 4
            trg_ops_list.append(x)
            trg_ops_np = np.array(trg_ops_list)
            trg_ops = torch.from_numpy(trg_ops_np).to(device)


        # import pudb
        # pudb.set_trace()

        # new_src_list = []
        # for i in src.cpu().numpy().tolist():
        #     i.pop(1)
        #     i.append(1)
        #     new_src_list.append(i)
        # new_src_ndarray = np.array(new_src_list)
        # new_src_tensor = torch.from_numpy(new_src_ndarray).to(device)
        # new_trg_list = []
        # for i in trg.cpu().numpy().tolist():
        #     i.pop(1)
        #     i.append(1)
        #     new_trg_list.append(i)
        # new_trg_ndarray = np.array(new_trg_list)
        # new_trg_tensor = torch.from_numpy(new_trg_ndarray).to(device)
        # src = new_src_tensor
        # trg = new_trg_tensor

        optimizer.zero_grad()

        output, _ = model(src, style_tok_list_tensor_enc, style_tok_list_tensor_dec, trg[:, :-1])
        output1, _ = model(src, style_tok_list_tensor_enc, style_tok_list_tensor_dec_ops, trg_ops[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]


        #########################################################################################
        senti_loss2 = None
        if(epoch>=0):
            senti_score_list = []
            senti_score_list2 = []
            for i, x in enumerate(output1.cpu().detach().numpy()):
                # single_tensor = torch.from_numpy(x)
                gen_trg_indexes = []
                for x_ind in x:
                    single_tensor = torch.from_numpy(x_ind)
                    gen_pred_token = single_tensor.argmax().item()
                    gen_trg_indexes.append(gen_pred_token)
                    if gen_pred_token == TRG.vocab.stoi[TRG.eos_token]:
                        break
                trg_tokens = [TRG.vocab.itos[i] for i in gen_trg_indexes]
                # if(epoch==25):
                #     print(post_processing(trg_tokens), flush=True)
                #     print('\n', flush=True)

                # import pudb
                # pudb.set_trace()
                #trn_sent = translate_sentence(post_processing(trg_tokens), SRC, TRG, model, device, is_st=True)
                senti_pred_score = sentiment.predict_sentiment(sentiment.model, sentiment.tokenizer,
                                                               post_processing(trg_tokens))
                senti_score_list.append(senti_pred_score)
                senti_score_list2.append([senti_pred_score for i in range(100)])

            #senti_pred = np.array(senti_score_list)
            senti_trg = np.array(senti_trg_list_ops)
            #senti_pred_tensor = torch.tensor(senti_pred, requires_grad=True)
            senti_trg_tensor = torch.tensor(senti_trg)
            #loss = nn.CrossEntropyLoss()
            loss = nn.BCELoss()
            #senti_loss = loss(senti_pred_tensor, senti_trg_tensor)

            senti_pred2 = np.array(senti_score_list)
            senti_pred_tensor2 = torch.tensor(senti_pred2, requires_grad=True)
            senti_trg_tensor = senti_trg_tensor.double()
            senti_loss2 = loss(senti_pred_tensor2, senti_trg_tensor)
        ################################################################################################


        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)
        new_loss = None
        actual_loss = None
        if(epoch>=0):
            #new_loss = loss + senti_loss2
            new_loss = loss
            # new_loss = None
            # if(loss_bool == True):
            #     new_loss = loss
            #     loss_bool = False
            # else:
            #     new_loss = senti_loss2
            #     loss_bool = True
            # print("Loss Testing", flush=True)
            # print("######################################################################################", flush=True)
            # print(loss.item(), flush=True)
            # print(senti_loss2.item(), flush=True)
            # print(new_loss.item(), flush=True)
            # print("######################################################################################", flush=True)
            # print("\n")
            new_loss.backward()
            actual_loss = new_loss
        else:
            loss.backward()
            actual_loss = loss

        list_translation_loss.append(loss.item())
        list_style_loss.append(senti_loss2.item())
        #loss_counter += 1
        # if (loss_counter % 1000 == 0):
        #     print(
        #         f'Translation Loss: {sum(list_translation_loss) / len(list_translation_loss)},Style Loss: {sum(list_style_loss) / len(list_style_loss)}',
        #         flush=True)
        #     list_translation_loss.clear()
        #     list_style_loss.clear()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_translation_loss += loss.item()
        epoch_style_loss += senti_loss2.item()
        epoch_loss += actual_loss.item()

    epoch_translation_loss_avg = epoch_translation_loss / len(iterator)
    epoch_style_loss_avg = epoch_style_loss / len(iterator)
    epoch_overall_loss = epoch_loss / len(iterator)

    return epoch_translation_loss_avg, epoch_style_loss_avg, epoch_overall_loss
#######################################################################################################################
import functools
def post_processing(text_list):
    repl_list = {'@@ ': '', '<eos>':''}
    text_str = ' '.join(text_list)
    return functools.reduce(lambda a, kv: a.replace(*kv), repl_list.items(), text_str)
#######################################################################################################################
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    #####################
    epoch_style_loss = 0
    epoch_translation_loss = 0
    loss_bool = True
    list_translation_loss = []
    list_style_loss = []
    loss_counter = 0
    #####################

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            ############################
            style_tok_list_enc = []
            style_tok_list_dec = []
            style_tok_list_dec_ops = []
            senti_trg_list = []
            senti_trg_list_ops = []
            ############################

            ####################################
            for i, x in enumerate(src.cpu().numpy()):
                # for j, k in enumerate(x):
                if (x[0] == 3):
                    senti_trg_list_ops.append(0)
                    style_tok_list_enc.append(pos_list_enc)
                    style_tok_list_dec.append(pos_list_dec)
                    style_tok_list_dec_ops.append(neg_list_dec)
                elif (x[0] == 4):
                    senti_trg_list_ops.append(1)
                    style_tok_list_enc.append(neg_list_enc)
                    style_tok_list_dec.append(neg_list_dec)
                    style_tok_list_dec_ops.append(pos_list_dec)
            style_tok_list_np_enc = np.array(style_tok_list_enc)
            style_tok_list_np_dec = np.array(style_tok_list_dec)
            style_tok_list_np_dec_ops = np.array(style_tok_list_dec_ops)
            style_tok_list_tensor_enc = torch.from_numpy(style_tok_list_np_enc).to(device)
            style_tok_list_tensor_dec = torch.from_numpy(style_tok_list_np_dec).to(device)
            style_tok_list_tensor_dec_ops = torch.from_numpy(style_tok_list_np_dec_ops).to(device)

            trg_copy = trg.detach().clone()
            trg_ops_list = []
            for i, x in enumerate(trg_copy.cpu().numpy()):
                if (x[0] == 3):
                    x[0] = 4
                elif (x[0] == 4):
                    x[0] = 3
                trg_ops_list.append(x)
                trg_ops_np = np.array(trg_ops_list)
                trg_ops = torch.from_numpy(trg_ops_np).to(device)
            ##########################################

            # new_src_list = []
            # for i in src.cpu().numpy().tolist():
            #     i.pop(1)
            #     i.append(1)
            #     new_src_list.append(i)
            # new_src_ndarray = np.array(new_src_list)
            # new_src_tensor = torch.from_numpy(new_src_ndarray).to(device)
            # new_trg_list = []
            # for i in trg.cpu().numpy().tolist():
            #     i.pop(1)
            #     i.append(1)
            #     new_trg_list.append(i)
            # new_trg_ndarray = np.array(new_trg_list)
            # new_trg_tensor = torch.from_numpy(new_trg_ndarray).to(device)
            # src = new_src_tensor
            # trg = new_trg_tensor

            #output, _ = model(src, style_tok_list_tensor_enc, style_tok_list_tensor_dec, trg[:, :-1])

            #######################################################
            output, _ = model(src, style_tok_list_tensor_enc, style_tok_list_tensor_dec, trg[:, :-1])
            output1, _ = model(src, style_tok_list_tensor_enc, style_tok_list_tensor_dec_ops, trg_ops[:, :-1])
            #######################################################

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            #########################################################################################
            senti_loss2 = None
            if (epoch >= 0):
                senti_score_list = []
                senti_score_list2 = []
                for i, x in enumerate(output1.cpu().detach().numpy()):
                    # single_tensor = torch.from_numpy(x)
                    gen_trg_indexes = []
                    for x_ind in x:
                        single_tensor = torch.from_numpy(x_ind)
                        gen_pred_token = single_tensor.argmax().item()
                        gen_trg_indexes.append(gen_pred_token)
                        if gen_pred_token == TRG.vocab.stoi[TRG.eos_token]:
                            break
                    trg_tokens = [TRG.vocab.itos[i] for i in gen_trg_indexes]
                    # if(epoch==25):
                    #     print(post_processing(trg_tokens), flush=True)
                    #     print('\n', flush=True)

                    # import pudb
                    # pudb.set_trace()
                    # trn_sent = translate_sentence(post_processing(trg_tokens), SRC, TRG, model, device, is_st=True)
                    senti_pred_score = sentiment.predict_sentiment(sentiment.model, sentiment.tokenizer,
                                                                   post_processing(trg_tokens))
                    senti_score_list.append(senti_pred_score)
                    senti_score_list2.append([senti_pred_score for i in range(100)])

                # senti_pred = np.array(senti_score_list)
                senti_trg = np.array(senti_trg_list_ops)
                # senti_pred_tensor = torch.tensor(senti_pred, requires_grad=True)
                senti_trg_tensor = torch.tensor(senti_trg)
                # loss = nn.CrossEntropyLoss()
                loss = nn.BCELoss()
                # senti_loss = loss(senti_pred_tensor, senti_trg_tensor)

                senti_pred2 = np.array(senti_score_list)
                senti_pred_tensor2 = torch.tensor(senti_pred2, requires_grad=True)
                senti_trg_tensor = senti_trg_tensor.double()
                senti_loss2 = loss(senti_pred_tensor2, senti_trg_tensor)
            ################################################################################################
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)
            new_loss = None
            actual_loss = None
            if (epoch >= 0):
                #new_loss = loss + senti_loss2
                new_loss = loss
                # new_loss = None
                # if(loss_bool == True):
                #     new_loss = loss
                #     loss_bool = False
                # else:
                #     new_loss = senti_loss2
                #     loss_bool = True
                # print("Loss Testing", flush=True)
                # print("######################################################################################", flush=True)
                # print(loss.item(), flush=True)
                # print(senti_loss2.item(), flush=True)
                # print(new_loss.item(), flush=True)
                # print("######################################################################################", flush=True)
                # print("\n")
                actual_loss = new_loss
            else:
                actual_loss = loss

            list_translation_loss.append(loss.item())
            list_style_loss.append(senti_loss2.item())
            # loss_counter += 1
            # if (loss_counter % 1000 == 0):
            #     print(
            #         f'Translation Loss: {sum(list_translation_loss) / len(list_translation_loss)},Style Loss: {sum(list_style_loss) / len(list_style_loss)}',
            #         flush=True)
            #     list_translation_loss.clear()
            #     list_style_loss.clear()

            epoch_translation_loss += loss.item()
            epoch_style_loss += senti_loss2.item()
            epoch_loss += actual_loss.item()

        epoch_translation_loss_avg = epoch_translation_loss / len(iterator)
        epoch_style_loss_avg = epoch_style_loss / len(iterator)
        epoch_overall_loss = epoch_loss / len(iterator)

        return epoch_translation_loss_avg, epoch_style_loss_avg, epoch_overall_loss
############################################################################################################################
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
#######################################################################################################################
N_EPOCHS = 50
CLIP = 1

best_valid_loss = float('inf')
valid_loss_before = float('inf')

early_stop_cnt = 0
early_stop_lookout = 5
early_stop=False

another_early_stop_cnt = 0
another_early_stop_lookout = 10
another_early_stop=False

best_epoch_no = 1

for epoch in range(N_EPOCHS):

    start_time = time.time()

    epoch_translation_loss_avg, epoch_style_loss_avg, epoch_overall_loss = train(model, train_iterator, optimizer, criterion, CLIP, epoch)
    #valid_loss = evaluate(model, valid_iterator, criterion)
    valid_translation_loss_avg, valid_style_loss_avg, valid_overall_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    #Early_Stop
    if valid_overall_loss < best_valid_loss or valid_overall_loss < valid_loss_before:
        early_stop_cnt = 0
        early_stop = False

    elif valid_overall_loss >= best_valid_loss or valid_overall_loss >= valid_loss_before:
        early_stop_cnt += 1
        early_stop = True

    # Another Early_Stop based on only best valid loss
    if epoch >= 20 :
        if valid_overall_loss < best_valid_loss:
            another_early_stop_cnt = 0
            another_early_stop = False

        elif valid_overall_loss >= best_valid_loss:
            another_early_stop_cnt += 1
            another_early_stop = True


    if valid_overall_loss < best_valid_loss:
        best_valid_loss = valid_overall_loss
        best_epoch_no = epoch

        torch.save(checkpoint, 'checkpoint.pt')

    valid_loss_before = valid_overall_loss


    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s', flush=True)
    print(f'\tTrain Translation Loss: {epoch_translation_loss_avg:.3f}', flush=True)
    print(f'\tTrain Style Loss: {epoch_style_loss_avg:.3f}', flush=True)
    print(f'\tTrain Overall Loss: {epoch_overall_loss:.3f} | Train PPL: {math.exp(epoch_overall_loss):7.3f}', flush=True)

    print(f'\t Val. Translation Loss: {valid_translation_loss_avg:.3f}', flush=True)
    print(f'\t Val. Style Loss: {valid_style_loss_avg:.3f}', flush=True)
    print(f'\t Val. Overall Loss: {valid_overall_loss:.3f} |  Val. PPL: {math.exp(valid_overall_loss):7.3f}', flush=True)
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
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#epoch = checkpoint['epoch']

#test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |', flush=True)
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
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, is_st=False):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    #tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    tokens = tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    ######################################################
    pos_list_enc = [3 for i in range(src_tensor.shape[1])]
    neg_list_enc = [4 for i in range(src_tensor.shape[1])]

    # pos_list_dec = [3 for i in range(src_tensor.shape[1]-1)]
    # neg_list_dec = [4 for i in range(src_tensor.shape[1]-1)]

    style_tok_list_enc = []
    if (src_field.vocab.stoi[tokens[0]] == 3):
        style_tok_list_enc.append(pos_list_enc)
    elif (src_field.vocab.stoi[tokens[0]] == 4):
        style_tok_list_enc.append(neg_list_enc)
    style_tok_list_np_enc = np.array(style_tok_list_enc)
    style_tok_list_tensor_enc = torch.from_numpy(style_tok_list_np_enc).to(device)
    style_embedd_enc = model.style_embedding(style_tok_list_tensor_enc)
    ##############################################################################

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, style_embedd_enc, src_mask)

    trg_indexes = None
    if(is_st == False):
        trg_indexes = [trg_field.vocab.stoi[tokens[0]]]
    else:
        if tokens[0] == '<p>':
            trg_indexes = [trg_field.vocab.stoi['<n>']]
        elif tokens[0] == '<n>':
            trg_indexes = [trg_field.vocab.stoi['<p>']]

    if (trg_field.vocab.stoi[tokens[0]] == 3):
        if (is_st == True):
            senti_trg_list_ops.append(0)
    elif (trg_field.vocab.stoi[tokens[0]] == 4):
        if (is_st == True):
            senti_trg_list_ops.append(1)

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        #########################################################
        pos_list_dec = [3 for i in range(trg_tensor.shape[1])]
        neg_list_dec = [4 for i in range(trg_tensor.shape[1])]
        style_tok_list_dec = []
        if (trg_field.vocab.stoi[tokens[0]] == 3):
            if (is_st == False):
                style_tok_list_dec.append(pos_list_dec)
            else:
                style_tok_list_dec.append(neg_list_dec)
        elif (trg_field.vocab.stoi[tokens[0]] == 4):
            if (is_st == False):
                style_tok_list_dec.append(neg_list_dec)
            else:
                style_tok_list_dec.append(pos_list_dec)
        style_tok_list_np_dec = np.array(style_tok_list_dec)
        style_tok_list_tensor_dec = torch.from_numpy(style_tok_list_np_dec).to(device)
        style_embedd_dec = model.style_embedding(style_tok_list_tensor_dec)
        #########################################################################################
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, style_embedd_dec, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention
######################################################################################################################
from torchtext.data.metrics import bleu_score
import nltk
senti_trg_list_ops = []
def masked_sent(sent_list):
    masked = snt_ev.mask_polarity(post_processing(sent_list))
    return nltk.word_tokenize(masked)
def calculate_bleu(data, src_field, trg_field, model, device, max_len=50, is_st=False):
    trgs = []
    masked_trgs = []

    pred_trgs = []
    masked_pred_trgs = []
    lengthy_idx = []
    for idx, datum in enumerate(data):
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        if len(src) < 100 and len(trg) < 100:
            pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len, is_st)

            # cut off <eos> token
            pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            masked_pred_trgs.append(masked_sent(pred_trg))
            trgs.append([trg])
            masked_trgs.append([masked_sent(trg)])
        else:
            lengthy_idx.append(idx)

    print(lengthy_idx)
    print(len(pred_trgs))
    return bleu_score(pred_trgs, trgs), bleu_score(masked_pred_trgs, masked_trgs)
# ######################################################################################################################
# b_score = calculate_bleu(test_data, SRC, TRG, model, device)
# print('\n')
# print(f'BLEU score without style transfer = {b_score*100:.2f}', flush=True)

b_score, masked_b_score = calculate_bleu(test_data, SRC, TRG, model, device, is_st=True)
print('\n')
print(f'BLEU score after style transfer = {b_score*100:.5f}', flush=True)
print(f'BLEU score after style transfer = {masked_b_score*100:.5f}', flush=True)


def evaluation(data):
    correct_count = 0
    lm_scores = []
    similarity_scores = []
    masked_similarity_scores = []

    for idx in range(1000):
        example_idx = idx

        src = vars(data.examples[example_idx])['src']
        trg = vars(data.examples[example_idx])['trg']


        predicted_trg_st, attention = translate_sentence(src, SRC, TRG, model, device, is_st=True)

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


print("Testing data", flush=True)
print('##############################################################################################################')
correct_count, lm_scores_mean, similarity_scores_mean, masked_similarity_scores_mean = evaluation(test_data)
print(correct_count)
print(lm_scores_mean)
print(correct_count)
print(lm_scores_mean)
print(similarity_scores_mean)
print(masked_similarity_scores_mean)

# #####################################################################################################################
# print("Style Transfer on Training data", flush=True)
# print('##############################################################################################################')
# for idx in range(50):
#     example_idx = idx
#
#     src = vars(train_data.examples[example_idx])['src']
#     trg = vars(train_data.examples[example_idx])['trg']
#
#     predicted_trg, attention = translate_sentence(src, SRC, TRG, model, device, is_st=True)
#
#     print(f'src = {post_processing(src)}', flush=True)
#     print(f'trg = {post_processing(trg)}', flush=True)
#     print(f'predicted trg = {post_processing(predicted_trg)}', flush=True)
#     print('\n', flush=True)
# print('##############################################################################################################')
# for idx in range(len(train_data.examples)-50, len(train_data.examples)):
#     example_idx = idx
#
#     src = vars(train_data.examples[example_idx])['src']
#     trg = vars(train_data.examples[example_idx])['trg']
#
#     predicted_trg, attention = translate_sentence(src, SRC, TRG, model, device, is_st=True)
#
#     print(f'src = {post_processing(src)}', flush=True)
#     print(f'trg = {post_processing(trg)}', flush=True)
#     print(f'predicted trg = {post_processing(predicted_trg)}', flush=True)
#     print('\n', flush=True)
# ######################################################################################################################
# print("Style Transfer on Testing data", flush=True)
# print('##############################################################################################################')
#
# senti_score_list = []
# # senti_trg_list = []
# senti_trg_list_ops = []
# # for i, x in enumerate(src.cpu().numpy()):
# #     # for j, k in enumerate(x):
# #     if (x[0] == 3):
# #         senti_trg_list.append(0)
# #         senti_trg_list_ops.append(1)
# #     elif (x[0] == 4):
# #         senti_trg_list.append(1)
# #         senti_trg_list_ops.append(0)
#
# for idx in range(len(test_data.examples)):
#     example_idx = idx
#
#     src = vars(test_data.examples[example_idx])['src']
#     trg = vars(test_data.examples[example_idx])['trg']
#
#     predicted_trg, attention = translate_sentence(src, SRC, TRG, model, device, is_st=True)
#
#     print(f'src = {post_processing(src)}', flush=True)
#     print(f'trg = {post_processing(trg)}', flush=True)
#     print(f'predicted trg = {post_processing(predicted_trg)}', flush=True)
#     senti_pred_score = sentiment.predict_sentiment(sentiment.model, sentiment.tokenizer,
#                                                    post_processing(predicted_trg))
#     # import pudb
#     # pudb.set_trace()
#
#     senti_score_list.append(senti_pred_score)
#     print('\n', flush=True)
#
# loss = nn.BCELoss()
#
# senti_trg = np.array(senti_trg_list_ops)
# senti_trg_tensor = torch.tensor(senti_trg)
#
# senti_pred2 = np.array(senti_score_list)
# senti_pred_tensor2 = torch.tensor(senti_pred2, requires_grad=True)
# senti_trg_tensor = senti_trg_tensor.double()
# senti_loss2 = loss(senti_pred_tensor2, senti_trg_tensor)
# print(f'Style Loss: {senti_loss2}', flush=True)
# ###########################################################################
# def binary_accuracy(preds, y):
#     rounded_preds = torch.round(preds)
#     correct = (rounded_preds == y).float()
#     acc = correct.sum() / len(correct)
#     return acc
#
# # import pudb
# # pudb.set_trace()
#
# style_acc = binary_accuracy(senti_pred_tensor2, senti_trg_tensor)
# print(f'Style Accuracy: {style_acc}', flush=True)