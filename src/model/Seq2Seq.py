import random
from gensim.models import Word2Vec
import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, input_seq, target_seq, initial_hidden=None):
        encoder_batch_size  = input_seq.size(1)
        decoder_batch_size = target_seq.size(1)
        initial_hidden = self.encoder.init_hidden(encoder_batch_size)
        #print("input_seq  size：", input_seq.shape)
        #print("target_seq size：", target_seq.shape)
        #print("initial hidden type & len：", type(initial_hidden), len(initial_hidden))
        #print("initial hidden size：", initial_hidden[0].shape, initial_hidden[1].shape)
        #print("initial hidden：", initial_hidden)

        encoder_output, encoder_hidden = self.encoder(input_seq, initial_hidden)
        #print("encoder hidden type & len：", type(encoder_hidden), len(encoder_hidden))
        #print("encoder hidden size：", encoder_hidden[0].shape, encoder_hidden[1].shape)
        #print("encoder hidden：", encoder_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden
        #print("\tencoder_hidden_h hidden type & len：", type(encoder_hidden_h), len(encoder_hidden_h))
        #print("\tencoder_hidden_h hidden size：", encoder_hidden_h[0].shape, encoder_hidden_h[1].shape)
        #print("\tencoder_hidden_c hidden type & len：", type(encoder_hidden_c), len(encoder_hidden_c))
        #print("\tencoder_hidden_c hidden size：", encoder_hidden_c[0].shape, encoder_hidden_c[1].shape)
        encoder_hidden_h_flat = encoder_hidden_h.view(encoder_hidden_h.size(0), -1)
        encoder_hidden_c_flat = encoder_hidden_c.view(encoder_hidden_c.size(0), -1)
        #print("encoder_hidden_h_flat hidden type & len：", type(encoder_hidden_h_flat), len(encoder_hidden_h_flat))
        #print("encoder_hidden_h_flat hidden size：", encoder_hidden_h_flat[0].shape, encoder_hidden_h_flat[1].shape)
        #print("encoder_hidden_c_flat hidden type & len：", type(encoder_hidden_c_flat), len(encoder_hidden_c_flat))
        #print("encoder_hidden_c_flat hidden size：", encoder_hidden_c_flat[0].shape, encoder_hidden_c_flat[1].shape)
        encoder_hidden_flat = torch.cat((encoder_hidden_h_flat, encoder_hidden_c_flat), dim=1)
        #print("\tencoder_hidden_flat hidden type & len：", type(encoder_hidden_flat), len(encoder_hidden_flat))
        #print("\tencoder_hidden_flat hidden size：", encoder_hidden_flat[0].shape, encoder_hidden_flat[1].shape)
        del initial_hidden,
        encoder_output, encoder_hidden,
        encoder_hidden_h, encoder_hidden_c,
        encoder_hidden_h_flat, encoder_hidden_c_flat
        
        hidden_mapping_layer = nn.Linear(
            in_features = encoder_hidden_flat.size(1),
            out_features = self.decoder.num_layers * decoder_batch_size * self.decoder.hidden_size)
        #print("hidden_mapping_layer in_features ：", self.encoder.num_layers, encoder_batch_size, self.encoder.hidden_size, self.encoder.num_layers * encoder_batch_size * self.encoder.hidden_size)
        #print("hidden_mapping_layer out_features：", self.decoder.num_layers, decoder_batch_size, self.decoder.hidden_size, self.decoder.num_layers * decoder_batch_size * self.decoder.hidden_size)
        
        mapped_hidden_flat = hidden_mapping_layer(encoder_hidden_flat)
        #print(" mapped_hidden_flat type & len：", type(mapped_hidden_flat), len(mapped_hidden_flat))
        #print(" mapped_hidden_flat size：", mapped_hidden_flat[0].shape, mapped_hidden_flat[1].shape
        mapped_hidden_h_flat, mapped_hidden_c_flat = mapped_hidden_flat.view(-1, 2, target_seq.size(1), self.decoder.hidden_size).chunk(2, dim=1)
        #print("\tmapped_hidden_h_flat type & len：", type(mapped_hidden_h_flat), len(mapped_hidden_h_flat))
        #print("\tmapped_hidden_h_flat size：", mapped_hidden_h_flat[0].shape, mapped_hidden_h_flat[1].shape)
        #print("\tmapped_hidden_c_flat type & len：", type(mapped_hidden_c_flat), len(mapped_hidden_c_flat))
        #print("\tmapped_hidden_c_flat size：", mapped_hidden_c_flat[0].shape, mapped_hidden_c_flat[1].shape)
        mapped_hidden_h = mapped_hidden_h_flat.view(self.decoder.num_layers, decoder_batch_size, self.decoder.hidden_size)
        mapped_hidden_c = mapped_hidden_c_flat.view(self.decoder.num_layers, decoder_batch_size, self.decoder.hidden_size)
        #print(" mapped_hidden_h type & len：", type(mapped_hidden_h), len(mapped_hidden_h))
        #print(" mapped_hidden_h size：", mapped_hidden_h[0].shape, mapped_hidden_h[1].shape)
        #print(" mapped_hidden_c type & len：", type(mapped_hidden_c), len(mapped_hidden_c))
        #print(" mapped_hidden_c size：", mapped_hidden_c[0].shape, mapped_hidden_c[1].shape)
        mapped_hidden = ( mapped_hidden_h, mapped_hidden_c )
        #print(" mapped_hidden type & len：", type(mapped_hidden), len(mapped_hidden))
        #print(" mapped_hidden size：", mapped_hidden[0].shape, mapped_hidden[1].shape)
        del hidden_mapping_layer, mapped_hidden_flat,
        mapped_hidden_h_flat, mapped_hidden_c_flat,
        mapped_hidden_h, mapped_hidden_c
        
        decoder_output, decoder_hidden = self.decoder(target_seq, mapped_hidden)
        #print("decoder hidden type & len：", type(decoder_hidden), len(decoder_hidden))
        #print("decoder hidden size：", decoder_hidden[0].shape, decoder_hidden[1].shape)
        #print("decoder hidden：", decoder_hidden)
        #print()
        return decoder_output

    def evaluate(self, input_seq, target_seq):
        
        embedded_input_seq  = self.encoder.embedding(input_seq)
        embedded_target_seq = self.decoder.embedding(target_seq)
        outputs = self.forward(embedded_input_seq.unsqueeze(0), embedded_target_seq.unsqueeze(0))
        
        return outputs
