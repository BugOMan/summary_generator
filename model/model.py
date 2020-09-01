#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
import config
from utils import timer, replace_oovs


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 rnn_drop: float = 0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            bidirectional=True,
                            dropout=rnn_drop,
                            batch_first=True)

#     @timer('encoder')
    def forward(self, x, decoder_embedding):
        """Define forward propagation for the endoer.

        Args:
            x (Tensor): The input samples as shape (batch_size, seq_len).
            decoder_embedding (torch.nn.modules): The input embedding layer from decoder
        Returns:
            output (Tensor):
                The output of lstm with shape
                (batch_size, seq_len, 2 * hidden_units).
            hidden (tuple):
                The hidden states of lstm (h_n, c_n).
                Each with shape (2, batch_size, hidden_units)
        """
        if config.weight_tying:
            embedded = decoder_embedding(x)
        else:
            embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        # Define feed-forward layers.
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)
        # wc for coverage feature
        self.wc = nn.Linear(1, 2*hidden_units, bias=False)
        self.v = nn.Linear(2*hidden_units, 1, bias=False)

#     @timer('attention')
    def forward(self,
                decoder_states,
                encoder_output,
                x_padding_masks,
                coverage_vector):
        """Define forward propagation for the attention network.

        Args:
            decoder_states (tuple):
                The hidden states from lstm (h_n, c_n) in the decoder,
                each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor):
                The output from the lstm in the decoder with
                shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).
            coverage_vector (Tensor):
                The coverage vector from last time step.
                with shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor):
                Dot products of attention weights and encoder hidden states.
                The shape is (batch_size, 2*hidden_units).
            attention_weights (Tensor): The shape is (batch_size, seq_length).
            coverage_vector (Tensor): The shape is (batch_size, seq_length).
        """
        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)
        # (batch_size, 1, 2*hidden_units)
        s_t = s_t.transpose(0, 1)
        # (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # calculate attention scores
        # Equation(1).
        # Wh h_* (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output.contiguous())
        # Ws s_t (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)
        # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features

        # Add coverage feature.
        if config.coverage:
            # Equation(11).
            coverage_features = self.wc(coverage_vector.unsqueeze(2))  # wc c
            att_inputs = att_inputs + coverage_features

        # (batch_size, seq_length, 1)
        score = self.v(torch.tanh(att_inputs))
        # Equation(2).
        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)
        attention_weights = attention_weights * x_padding_masks
        # Normalize attention weights after excluding padded positions.
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor
        # Equation(3).
        # (batch_size, 1, 2*hidden_units)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        # Update coverage vector.
        if config.coverage:
            # Equation(10).
            coverage_vector = coverage_vector + attention_weights

        return context_vector, attention_weights, coverage_vector


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 enc_hidden_size=None,
                 is_cuda=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)
        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

#     @timer('decoder')
    def forward(self, x_t, decoder_states, context_vector):
        """Define forward propagation for the decoder.

        Args:
            x_t (Tensor):
                The input of the decoder x_t of shape (batch_size, 1).
            decoder_states (tuple):
                The hidden states(h_n, c_n) of the decoder from last time step.
                The shapes are (1, batch_size, hidden_units) for each.
            context_vector (Tensor):
                The context vector from the attention network
                of shape (batch_size,2*hidden_units).

        Returns:
            p_vocab (Tensor):
                The vocabulary distribution of shape (batch_size, vocab_size).
            docoder_states (tuple):
                The lstm states in the decoder.
                The shapes are (1, batch_size, hidden_units) for each.
            p_gen (Tensor):
                The generation probabilities of shape (batch_size, 1).
        """
        decoder_emb = self.embedding(x_t)

        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # Equation(4)
        # concatenate context vector and decoder state
        # (batch_size, 3*hidden_units)
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat(
            [decoder_output,
             context_vector],
            dim=-1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        if config.weight_tying:
            FF2_out = torch.mm(FF1_out, torch.t(self.embedding.weight))
        else:
            FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        p_gen = None
        if config.pointer:
            # Equation(8).
            # Calculate p_gen.
            x_gen = torch.cat([
                context_vector,
                s_t.squeeze(0),
                decoder_emb.squeeze(1)
            ],
                              dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    """
    Since the encoder has a bidirectional LSTM layer while the decoder has a
    unidirectional LSTM layer, we add this module to reduce the hidden states
    output by the encoder (merge two directions) before input the hidden states
    nto the decoder.
    """
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        """The forward propagation of reduce state module.

        Args:
            hidden (tuple):
                Hidden states of encoder,
                each with shape (2, batch_size, hidden_units).

        Returns:
            tuple:
                Reduced hidden states,
                each with shape (1, batch_size, hidden_units).
        """
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)


class PGN(nn.Module):
    def __init__(
            self,
            v
    ):
        super(PGN, self).__init__()
        self.v = v
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(
            len(v),
            config.embed_size,
            config.hidden_size,
        )
        self.decoder = Decoder(len(v),
                               config.embed_size,
                               config.hidden_size,
                               )
        self.reduce_state = ReduceState()

    def load_model(self):

        if (os.path.exists(config.encoder_save_name)):
            print('Loading model: ', config.encoder_save_name)
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)

        elif config.fine_tune:
            print('Loading model: ', '../saved_model/pgn/encoder.pt')
            self.encoder = torch.load('../saved_model/pgn/encoder.pt')
            self.decoder = torch.load('../saved_model/pgn/decoder.pt')
            self.attention = torch.load('../saved_model/pgn/attention.pt')
            self.reduce_state = torch.load('../saved_model/pgn/reduce_state.pt')

#     @timer('final dist')
    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights,
                               max_oov):
        """Calculate the final distribution for the model.

        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.

        Returns:
            final_distribution (Tensor):
            The final distribution over the extended vocabualary.
            The shape is (batch_size, )
        """

        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        # Clip the probabilities.
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        # Get the weighted probabilities.
        # Equation (9).
        p_vocab_weighted = p_gen * p_vocab
        # (batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        # Get the extended-vocab probability distribution
        # extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)
        # (batch_size, extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # Add the attention weights to the corresponding vocab positions.
        final_distribution = \
            p_vocab_extended.scatter_add_(dim=1,
                                          index=x,
                                          src=attention_weighted)

        return final_distribution

#     @timer('model forward')
    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):
        """Define the forward propagation for the seq2seq model.

        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (Tensor):
                The numbers of out-of-vocabulary words for samples in this batch.
            batch (int): The number of the current batch.
            num_batches(int): Number of batches in the epoch.
            teacher_forcing(bool): teacher_forcing or not

        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """

        x_copy = replace_oovs(x, self.v)
        x_padding_masks = torch.ne(x, 0).byte().float()
        encoder_output, encoder_states = self.encoder(x_copy, self.decoder.embedding)
        # Reduce encoder hidden states.
        decoder_states = self.reduce_state(encoder_states)
        # Initialize coverage vector.
        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)
        # Calculate loss for every step.
        step_losses = []
        x_t = y[:, 0]
        for t in range(y.shape[1]-1):

            # Do teacher forcing.
            if teacher_forcing:
                x_t = y[:, t]
            x_t = replace_oovs(x_t, self.v)

            y_t = y[:, t+1]
            # Get context vector from the attention network.
            context_vector, attention_weights, coverage_vector = \
                self.attention(decoder_states,
                               encoder_output,
                               x_padding_masks,
                               coverage_vector)
            # Get vocab distribution and hidden states from the decoder.
            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1),
                                                          decoder_states,
                                                          context_vector)

            final_dist = self.get_final_distribution(x,
                                                     p_gen,
                                                     p_vocab,
                                                     attention_weights,
                                                     torch.max(len_oovs))

            # Get the probabilities predict by the model for target tokens.
            if not config.pointer:
                y_t = replace_oovs(y_t, self.v)
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(y_t, 0).byte()
            # Equation(6)
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = -torch.log(target_probs + config.eps)

            if config.coverage:
                # Equation(12)
                # Add coverage loss.
                ct_min = torch.min(attention_weights, coverage_vector)
                # Equation(13)
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss + config.LAMBDA * cov_loss

            mask = mask.float()
            loss = loss * mask

            step_losses.append(loss)
        # Equation(7)
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss

