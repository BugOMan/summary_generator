#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

# General
hidden_size = 512
dec_hidden_size = 512
embed_size = 300
pointer = True

# Data
max_vocab_size = 20000
embed_file = None  # use pre-trained embeddings
data_path: str = '../files/train.txt'
val_data_path = '../files/dev.txt'
test_data_path = '../files/test.txt'
stop_word_file: str = '../files/HIT_stop_words.txt'
max_src_len = 300  # exclusive of special tokens such as EOS
max_tgt_len = 100  # exclusive of special tokens such as EOS
truncate_src = True
truncate_tgt = True
min_dec_steps = 30
max_dec_steps = 100
enc_rnn_dropout = 0.5
enc_attn = True
dec_attn = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0


# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 1
batch_size = 32
coverage = True
fine_tune = False
max_grad_norm = 2.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1

if pointer:
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    else:
        model_name = 'pgn'

else:
    model_name = 'baseline'

encoder_save_name = '../saved_model/' + model_name + '/encoder.pt'
decoder_save_name = '../saved_model/' + model_name + '/decoder.pt'
attention_save_name = '../saved_model/' + model_name + '/attention.pt'
reduce_state_save_name = '../saved_model/' + model_name + '/reduce_state.pt'
losses_path = '../saved_model/' + model_name + '/val_losses.pkl'
log_path = '../runs/' + model_name


# Beam search
beam_size = 3
alpha = 0.2
beta = 0.2
gamma = 0.6
