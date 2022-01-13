import numpy as np
import torch
from torch.autograd import Variable as V
import time
import torch.utils.data as Data
from torch import nn
from torch.nn import Module
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder
from temporal_decoder_layer import temporal_decoder_layer
from torch.nn import TransformerDecoder
from torch.nn import LayerNorm
from torch.nn import BatchNorm1d
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.nn import Dropout
from torch.nn.init import xavier_uniform_
import json


class STForecasting(Module):
    def __init__(self, out_channel = 8, kernal_size=3, stride = 1, d_model=64, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=1, dim_feedforward=128,
                 dropout=0.1, dataset = './taxi_data.json',
                 activation="relu"):
        super(STForecasting, self).__init__()
        print('num_head is ' + str(nhead))

        self.config = json.load(open(dataset, "r"))

        self.local_context_len = self.config['local_context_len']
        self.in_conv = nn.Conv1d(1, out_channel, kernal_size, stride)
        self.out_conv = nn.Conv1d(1,out_channel,kernal_size, stride)
        self.embedding_in = Linear(out_channel * (self.local_context_len - (kernal_size - 1)), d_model)
        self.embedding_out = Linear(out_channel * (self.local_context_len - (kernal_size - 1)), d_model)
        self.embedding = Linear(2 * out_channel * (self.local_context_len - (kernal_size - 1)), d_model)
        self.spatial_embedding_layer = Linear(200, d_model)
        self.temporal_embedding_layer = Linear(48,d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)



        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        decoder_layer_in = temporal_decoder_layer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm_in = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer_in, num_decoder_layers, decoder_norm_in)

        self.conv_norm_in = BatchNorm1d(64)
        self.conv_norm_in = BatchNorm1d(64)


        self.linear5 = Linear(d_model, 16)
        self.linear6 = Linear(16, 2)

        self.d_model = d_model
        self.nhead = nhead
        self.src_mask = None
        self.tgt_mask = None
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
                #kaiming_uniform_(p,nonlinearity='relu')

    def forward(self, src, spatial_embedding, temporal_embedding,random_mask):

        feature_num = self.local_context_len
        slot_num = self.config["trend_num"] + self.config["daily_period_num"] - 1


        in_src = src[:, :, 0:feature_num].reshape(src.shape[0] * src.shape[1], 1,feature_num)
        out_src = src[:, :, feature_num:feature_num*2].reshape(src.shape[0] * src.shape[1], 1, feature_num)
        in_src_result = relu(self.in_conv(in_src))
        out_src_result = relu(self.out_conv(out_src))


        

        in_result = in_src_result.reshape(src.shape[0],src.shape[1],in_src_result.shape[1] * in_src_result.shape[2])
        out_result = out_src_result.reshape(src.shape[0],src.shape[1],in_src_result.shape[1] * in_src_result.shape[2])
        result = torch.cat((in_result,out_result),2)
        result = relu(self.embedding(result))
        spatial_embedding = relu(self.spatial_embedding_layer(spatial_embedding))
        temporal_embedding = relu(self.temporal_embedding_layer(temporal_embedding))
        result = result + spatial_embedding
        #result = self.norm1(result)
        result = result + temporal_embedding
        #result = self.norm2(result)
        result = self.encoder(result,random_mask)
        
        tgt = result.reshape(1, result.shape[0] * result.shape[1], self.d_model)
        
        flow = tgt

        for i in range(1, slot_num):
            #print(i)
            in_src = src[:, :, feature_num * 2 * i: feature_num * 2 * i + feature_num].reshape(src.shape[0] * src.shape[1], 1,feature_num)
            out_src = src[:, :, feature_num * 2 * i + feature_num : feature_num * 2 * (i + 1)].reshape(src.shape[0] * src.shape[1], 1, feature_num)
            in_src_result = relu(self.in_conv(in_src))
            out_src_result = relu(self.out_conv(out_src)) 

           

            in_result = in_src_result.reshape(src.shape[0], src.shape[1],in_src_result.shape[1] * in_src_result.shape[2])
            out_result = out_src_result.reshape(src.shape[0], src.shape[1],in_src_result.shape[1] * in_src_result.shape[2])
            
            result = torch.cat((in_result, out_result), 2)
            result = relu(self.embedding(result))
            
            result = result + spatial_embedding
            
            result = result + temporal_embedding
            
            result = self.encoder(result,random_mask)
            result = result.reshape(1, result.shape[0] * result.shape[1], self.d_model)
            flow = torch.cat((flow,result),0)

            

        
        tgt_tmp = self.decoder(tgt,flow)
        tgt = tgt + tgt_tmp
        tgt = tgt.reshape(200,int(tgt.shape[1]/200),self.d_model)
        value = self.linear6((relu(self.linear5(tgt))))
        return value