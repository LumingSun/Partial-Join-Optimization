# embedding queries to vectors using auto-encoder

# %%
import os
from d2l import torch as d2l
import torch
from torch import nn
from net import TransformerEncoder, TransformerDecoder, EncoderDecoder
import math
import numpy as np

def train_seq2seq(net, data_iter, val_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型
    Defined in :numref:`sec_seq2seq_decoder`"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = d2l.MaskedSoftmaxCELoss()
    
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',legend=["train"],
                     xlim=[10, num_epochs])
    val_animator = d2l.Animator(xlabel='epoch', ylabel='loss',legend=["validation"],
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        val_metric = d2l.Accumulator(2)
        for batch,val_batch in zip(data_iter,val_iter):
            net.train()
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()	# 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            
            
            net.eval()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in val_batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            val_metric.add(l.sum(), num_tokens)
                
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
            val_animator.add(epoch + 1, (val_metric[0] / val_metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
    print(f'validation loss {val_metric[0] / val_metric[1]:.3f}')

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
    num_steps = max([len(line) for line in tokens])
    # num_steps = 200
    features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in tokens])
    valid_len = d2l.reduce_sum(
        d2l.astype(features != vocab['<pad>'], d2l.int32), 1)
    data_arrays = (features, valid_len, features, valid_len)
    data_iter = d2l.load_array((data_arrays), batch_size)
    return data_iter, vocab, num_steps

def get_encoder_result(net, query, src_vocab, num_steps, device):
    # num_steps = len(d2l.tokenize([query],token="word")[0]) + 1
    net.eval()
    src_tokens = src_vocab[query.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # print(enc_outputs)
    # print(enc_outputs.size())
    # print(num_steps)
    return enc_outputs
# %% 
# sql_dir = "../data/bao_sample_queries"
sql_dir = "../data/join-order-benchmark"  # len(vocab) = 616
data = read_queries(sql_dir)


# %%
num_hiddens, num_layers, dropout, batch_size  = 32, 2, 0.1, 16
# num_steps = 10
lr, num_epochs, device = 0.005, 500, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = num_hiddens, 64, 4
key_size, query_size, value_size = num_hiddens, num_hiddens, num_hiddens
norm_shape = [num_hiddens]

data_iter, vocab, num_steps = load_query_data(data,batch_size=batch_size)
val_iter, vocab, num_steps = load_query_data(data,batch_size=int(batch_size))

# %%
encoder = TransformerEncoder(
    len(vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, data_iter, lr, num_epochs, vocab, device)
# train_seq2seq(net, data_iter, val_iter, lr, num_epochs, vocab, device)
# %%
query_encodings = []
sql_files = sorted(os.listdir(sql_dir))
for file in sql_files:
    print(file)
    with open(os.path.join(sql_dir,file),"r") as f:
        query = f.read().replace("\n"," ")
    # num_steps = len(d2l.tokenize([query],token="word")[0]) + 2
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, query, vocab, vocab, num_steps, device, True)
    # print(d2l.bleu(translation, 
                #    " ".join(d2l.tokenize([query],token="word")[0]), k=2))
    query_encoding = get_encoder_result(net,query,vocab,num_steps,device)
    query_encodings.append(query_encoding.detach().cpu().numpy().flatten().tolist())
    # print(translation)    
    # break
# %%
with open("../result/embeded_query_id.tsv","w") as f:
    f.writelines([each[:-5]+"\n" for each in sql_files])
# %%

query_encodings_tsv = []
for each in query_encodings:
    query_encodings_tsv.append("\t".join([f'{every:.5f}' for every in each]))
with open("../result/embeded_queries.tsv","w") as f:
    f.writelines("\n".join(query_encodings_tsv))
# %%
