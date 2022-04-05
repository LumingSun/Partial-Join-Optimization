# %%
import torch
from sequitur.models import LINEAR_AE,CONV_LSTM_AE,LSTM_AE
from sequitur import quick_train
import os
from d2l import torch as d2l


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
    features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in tokens])
    valid_len = d2l.reduce_sum(
        d2l.astype(features != vocab['<pad>'], d2l.int32), 1)
    data_arrays = (features, valid_len, features, valid_len)
    data_iter = d2l.load_array((data_arrays), batch_size, is_train=False)
    return data_iter, vocab, num_steps
# %%
sql_dir = "../data/join-order-benchmark"  # len(vocab) = 616
data = read_queries(sql_dir)
data_iter, vocab, num_steps = load_query_data(data,batch_size=1)
# %%
data_set = []
for x,_,_,_ in data_iter:
    # print(x)
    x = x.view((1,-1))
    x = x.type(torch.float32)
    data_set.append(x)
    # break
# x = torch.cat(data_set, dim=0) 
# %%
model = LSTM_AE(
  input_dim=num_steps,
  encoding_dim=16,
  h_dims=[64],
  h_activ=None,
  out_activ=None
)

encoder, decoder, _, _ = quick_train(LSTM_AE, data_set, epochs=500, lr=0.01,
                                     verbose=True,encoding_dim=16, denoise=True)



# x = torch.cat(data_set, dim=0) 
# x = x.type(torch.float)
# z = model.encoder(x) 
# x_prime = model.decoder(z, seq_len=len(data_set)) 

# %%
