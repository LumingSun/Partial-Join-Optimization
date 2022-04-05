# %%
import os
from d2l import torch as d2l
import torch
from torch import nn
from net import TransformerEncoder, TransformerDecoder, EncoderDecoder
import math


def read_queries(sql_dir):
    data = []
    sql_files = sorted(os.listdir(sql_dir))
    for file in sql_files:
        with open(os.path.join(sql_dir,file),"r") as f:
            query = f.read().replace("\n"," ")
            data.append(query)
    return data


def load_query_data(data,batch_size):
    tokens = d2l.tokenize(data,token="word")
    vocab = d2l.Vocab(tokens,reserved_tokens=['<pad>'])
    num_steps = max([len(line) for line in tokens])+3
    # num_steps = 200
    features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in tokens])
    valid_len = d2l.reduce_sum(
        d2l.astype(features != vocab['<pad>'], d2l.int32), 1)
    data_arrays = (features, valid_len, features, valid_len)
    data_iter = d2l.load_array((data_arrays), batch_size)
    return data_iter, vocab, num_steps


def get_encoder_result(net, query, src_vocab,  device):
    num_steps = len(d2l.tokenize([query],token="word")[0]) + 1
    net.eval()
    src_tokens = src_vocab[query.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    print(enc_outputs)
    print(enc_outputs.size())
    print(num_steps)
    return enc_outputs


sql_files = sorted(os.listdir(sql_dir))
for file in sql_files:
    print(file)
    with open(os.path.join(sql_dir,file),"r") as f:
        query = f.read().replace("\n"," ")
    num_steps = len(d2l.tokenize([query],token="word")[0]) + 2
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, query, vocab, vocab, num_steps, device, True)
    print(d2l.bleu(translation, 
                   " ".join(d2l.tokenize([query],token="word")[0]), k=2))
    # print(translation)    
    # break