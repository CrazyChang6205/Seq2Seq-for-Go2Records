import os
import sys
import gc

import random
from gensim.models import Word2Vec
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    """
    此這版本的 LSTM of Encoder/Decoder 輸入已改為多個詞彙逐一輸入，如：[詞索引]、[詞索引]、...、[詞索引] (逐一輸入)
    此模型為 more-more 的結構，修改內容為 Encoder/Decoder forward
    """ 
    def forward(self, input_seq, target_seq, initial_hidden=None):
        encoder_batch_size  = input_seq.size(0)
        
        initial_hidden = self.encoder.init_hidden(encoder_batch_size)
        
        _ , encoder_hidden = self.encoder(input_seq, initial_hidden)

        output_seq, decoder_hidden = self.decoder(target_seq, encoder_hidden)

        return output_seq, decoder_hidden

    def evaluate(self, input_seq, outputs_vocab_MAX=50):
        encoder_batch_size  = input_seq.size(0)
        initial_hidden = self.encoder.init_hidden(encoder_batch_size)
        
        _ , encoder_hidden = self.encoder(input_seq, initial_hidden)

        output_seq = self.decoder.evaluate(encoder_hidden, outputs_vocab_MAX)
        
        return output_seq


class Seq2Seq_old_1(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    """
    在這版本的 forward 中輸入 LSTM of Encoder/Decoder 的是一串完整句子，如：[詞索引, 詞索引, ..., 詞索引] (單筆輸入)
    此模型為 one-one 的結構，不符合目前題目要求 more-more，要求輸入為多個詞彙逐一輸入，如：[詞索引]、[詞索引]、...、[詞索引] (逐一輸入)
    """    
    def forward(self, input_seq, target_seq, initial_hidden=None):
        encoder_batch_size  = input_seq.size(1)
        decoder_batch_size = target_seq.size(1)
        initial_hidden = self.encoder.init_hidden(encoder_batch_size)

        encoder_output, encoder_hidden = self.encoder(input_seq, initial_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden
        encoder_hidden_h_flat = encoder_hidden_h.view(encoder_hidden_h.size(0), -1)
        encoder_hidden_c_flat = encoder_hidden_c.view(encoder_hidden_c.size(0), -1)
        encoder_hidden_flat = torch.cat((encoder_hidden_h_flat, encoder_hidden_c_flat), dim=1)
        del initial_hidden, encoder_output, encoder_hidden, encoder_hidden_h, encoder_hidden_c, encoder_hidden_h_flat, encoder_hidden_c_flat
        
        hidden_mapping_layer = nn.Linear(
            in_features = encoder_hidden_flat.size(1),
            out_features = self.decoder.num_layers * decoder_batch_size * self.decoder.hidden_size)
        
        mapped_hidden_flat = hidden_mapping_layer(encoder_hidden_flat)
        mapped_hidden_h_flat, mapped_hidden_c_flat = mapped_hidden_flat.view(-1, 2, target_seq.size(1), self.decoder.hidden_size).chunk(2, dim=1)
        mapped_hidden_h = mapped_hidden_h_flat.view(self.decoder.num_layers, decoder_batch_size, self.decoder.hidden_size)
        mapped_hidden_c = mapped_hidden_c_flat.view(self.decoder.num_layers, decoder_batch_size, self.decoder.hidden_size)
        mapped_hidden = ( mapped_hidden_h, mapped_hidden_c )
        del hidden_mapping_layer, mapped_hidden_flat, mapped_hidden_h_flat, mapped_hidden_c_flat, mapped_hidden_h, mapped_hidden_c
        
        decoder_output, decoder_hidden = self.decoder(target_seq, mapped_hidden)

        return decoder_output