import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import deepwave
import pandas as pd
import torch.optim as optim
import itertools
import torch.utils
import torch.utils.data
import transformers
import time
import os
import torch.nn as nn
import math

from seismiclip.vit import *
from seismiclip.clip import *
from seismiclip.plots import *
from seismiclip.utils import *

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from transformers import BertConfig, BertForMaskedLM
from argparse import ArgumentParser
from radam import RAdam
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from typing import Optional, Tuple

def plot_square_images(array, height, width=4, name=None, save_dir=None, unit=None, transpose=False, **kwargs):
    
    fig, ax = plt.subplots(
        height, width, sharex=True, sharey=True, 
        figsize=(int(1.5*height), int(1.5*height)), layout="constrained"
    )

    axs = ax.ravel()
    
    idx = 0
    for i in range(width*height):
        if transpose:
            im = axs[idx].imshow(array[i,:,:].T, aspect='auto', **kwargs)
        else:
            im = axs[idx].imshow(array[i,:,:], aspect='auto', **kwargs)
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        idx+=1
        
    fig.supxlabel('Lateral Location (km)', fontsize=20)
    fig.supylabel('Depth (km)', fontsize=20)
        
    cbar = fig.colorbar(im, ax=ax[:,-1])
    cbar.set_label('km/s', size=20)
    cbar.ax.tick_params(labelsize=20) 
    if name is not None:
        plt.savefig(save_dir+'/'+name)
        
    plt.show()
    
def expand_array(array, sigmas):
    num_smoothings = len(sigmas)
    expanded_array = np.zeros((array.shape[0] + num_smoothings * array.shape[0], array.shape[1], array.shape[2]))

    expanded_array[::num_smoothings + 1] = array

    for i in range(num_smoothings):
        smoothed_image = gaussian_filter(array, sigma=[0,sigmas[i],sigmas[i]])
        expanded_array[i+1::num_smoothings + 1] = smoothed_image

    return expanded_array

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SeismicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['inputs_embeds'])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()
        
    def forward(self, x):
        return 0

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.pe[:x.size(1)]
        return self.dropout(x)
    
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)

        self.position_embeddings = PositionalEncoding(d_model=config.hidden_size,
                                                        max_len=config.max_position_embeddings)
        if config.add_alibi:
            self.position_embeddings.pe = torch.zeros(config.max_position_embeddings, 1, config.hidden_size)
        
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        
    def forward(self, inputs_embeds, input_ids=None, position_ids=None, token_type_ids=None,
               past_key_values_length=None):
        embeddings = self.word_embeddings(inputs_embeds)
        position_ids = self.position_ids
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings += position_embeddings.swapaxes(0, 1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output

class PreLNBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states
    
    
class PreLNBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states
    
class PreLNBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self = transformers.models.bert.modeling_bert.BertSelfAttention(config)
        self.output = transformers.models.bert.modeling_bert.BertSelfOutput(config)
        self.pruned_heads = set()
        
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, dim=1)
        
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
    
class PreLNBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = transformers.activations.ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
            
    def forward(self, hidden_states):
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class DenoisingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output

class VelpredHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vel_size)
        self.predictions.decoder = Identity()
        self.vel_min = config.vel_min
        self.vel_max = config.vel_max
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        output = torch.mean(output[:, :1, :], dim=1)
        output = self.vel_min + (output + 1) * (self.vel_max - self.vel_min) * 0.5
        
        return output

class LowFreqHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output

class DenseSynthesizerHead1(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.dense_synth_act == "relu":
            self.act_fn = nn.ReLU()
        elif config.dense_synth_act == "gelu":
            self.act_fn = nn.GELU()
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            self.act_fn,
            nn.Linear(config.hidden_size, config.max_length)
        )

    def forward(self, x):
        output = self.dense(x)

        return output
    
class DenseSynthesizerHead2(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.dense_synth_act == "relu":
            self.act_fn = nn.ReLU()
        elif config.dense_synth_act == "gelu":
            self.act_fn = nn.GELU()
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.max_length),
            self.act_fn,
            nn.Linear(config.max_length, config.max_length)
        )

    def forward(self, x):
        output = self.dense(x)

        return output

class RandomSynthesizerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.fixed:
            self.attention = nn.Parameter(torch.empty(config.num_attention_heads, config.max_length, config.max_length), requires_grad=False)
            # val1 = torch.ones(config.max_length - 1) * 0.5
            # val1[0] = 1
            # val2 = torch.ones(config.max_length - 1) * 0.5
            # val2[-1] = 1
            # self.attention = torch.diag(val1, 1) + torch.diag(val2, -1)
        else:
            self.attention = nn.Parameter(torch.empty(config.num_attention_heads, config.max_length, config.max_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)

    def forward(self):
        output = self.attention

        return output

class FactorizedRandomSynthesizerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fixed = config.fixed

        self.query_fc = nn.Parameter(torch.empty(config.num_attention_heads, config.max_length, config.k), requires_grad=True)
        nn.init.xavier_uniform_(self.query_fc)
        if not self.fixed:
            self.key_fc = nn.Parameter(torch.empty(config.num_attention_heads, config.max_length, config.k), requires_grad=True)
            nn.init.xavier_uniform_(self.key_fc)

    def forward(self):
        if not self.fixed:
            output = torch.einsum('hnk,hmk->hnm', self.query_fc, self.key_fc)
        else:
            output = torch.einsum('hnk,hmk->hnm', self.query_fc, self.query_fc)

        return output

class URPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.max_length = config.max_length
        
        self.urpe_weight_ = nn.Parameter(torch.ones(self.num_attention_heads, 2 * self.max_length), requires_grad=True)
    
    def forward(self, attention_probs):
        def toeplitz(c, r):
            vals = torch.cat((r, c[1:].flip(0)))
            shape = r.shape[0], r.shape[-1], c.shape[-1]
            i, j, k = torch.ones(*shape).nonzero().T
            return vals[i, k-j].reshape(*shape)
        
        self.urpe_weight = toeplitz(self.urpe_weight_[:, :self.max_length], self.urpe_weight_[:, self.max_length:])
        
        attention_probs = torch.mul(attention_probs, self.urpe_weight.to(attention_probs.device))
        
        return attention_probs

class LinearBiases(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.alibi_type = config.alibi_type

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:                       
                closest_power_of_2 = 2**math.floor(math.log2(n)) 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

        def fill_with_neg_inf(t):
            """FP16-compatible function that fills a tensor with -inf."""
            return t.float().fill_(float("-inf")).type_as(t)

        self.register_buffer("context_position", torch.arange(config.max_length)[:, None])
        self.register_buffer("memory_position", torch.arange(config.max_length)[None, :])
        self.register_buffer("relative_position", self.memory_position - self.context_position, persistent=False)

        if config.alibi_type == "sym":
            self.relative_position = torch.abs(self.relative_position).unsqueeze(0).expand(config.num_attention_heads, -1,-1)
            self.slopes = torch.Tensor(get_slopes(config.num_attention_heads))*-1
            self.bias = self.slopes.unsqueeze(1).unsqueeze(1) * self.relative_position
            self.bias = self.bias.view(1, config.num_attention_heads, config.max_length, config.max_length)
        elif config.alibi_type == "nosym_mask":
            self.relative_position = torch.abs(self.relative_position).unsqueeze(0).expand(config.num_attention_heads//2, -1,-1)
            self._future_mask_right = torch.triu(fill_with_neg_inf(torch.zeros([config.max_length, config.max_length])), 1).unsqueeze(0).repeat(config.num_attention_heads//2, 1, 1)
            self._future_mask_left = torch.tril(fill_with_neg_inf(torch.zeros([config.max_length, config.max_length])), -1).unsqueeze(0).repeat(config.num_attention_heads//2, 1, 1)
            self.nonsym_mask = torch.cat((self._future_mask_right, self._future_mask_left), dim = 0).unsqueeze(0)
            self.slopes = torch.Tensor(get_slopes(config.num_attention_heads//2))*-1
            self.bias = self.slopes.unsqueeze(1).unsqueeze(1) * self.relative_position
            self.bias = self.bias.view(1, config.num_attention_heads//2, config.max_length, config.max_length)
            self.bias = self.bias.repeat(1, 2, 1, 1)
        elif config.alibi_type == "nosym":
            self.relative_position = torch.abs(self.relative_position).unsqueeze(0).expand(config.num_attention_heads, -1,-1)
            if config.fixed_slopes:
                self.slopes_left = nn.Parameter(torch.empty(config.num_attention_heads), requires_grad=False)
                self.slopes_right = nn.Parameter(torch.empty(config.num_attention_heads), requires_grad=False)
            else:
                self.slopes_left = nn.Parameter(torch.empty(config.num_attention_heads), requires_grad=True)
                self.slopes_right = nn.Parameter(torch.empty(config.num_attention_heads), requires_grad=True)
            nn.init.normal_(self.slopes_left, -2, 1)
            nn.init.normal_(self.slopes_right, -2, 1)

    def forward(self, attention_scores):
        if self.alibi_type == "nosym":
            alibi_left = self.slopes_left.unsqueeze(1).unsqueeze(1) * self.relative_position
            alibi_right = self.slopes_right.unsqueeze(1).unsqueeze(1) * self.relative_position
            self.bias = torch.triu(alibi_right) + torch.tril(alibi_left)

        output = attention_scores + self.bias.repeat(attention_scores.size()[0], 1, 1, 1).to(attention_scores.device)

        return output

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.max_length = config.max_length
        self.attention_type = config.attention_type
        self.add_alibi = config.add_alibi
        self.alibi_type = config.alibi_type
        self.fixed_slopes = config.fixed_slopes
        self.add_urpe = config.add_urpe

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        if self.attention_type not in ["default", "default_fcrand"]:
            for params in self.query.parameters():
                 params.requires_grad = False
            for params in self.key.parameters():
                 params.requires_grad = False
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.attention_type == "dense_synth1":
            self.head = nn.ModuleList([DenseSynthesizerHead1(config) for _ in range(config.num_attention_heads)])
        elif self.attention_type == "dense_synth2":
            self.head = nn.ModuleList([DenseSynthesizerHead2(config) for _ in range(config.num_attention_heads)])
        elif self.attention_type == "rand_synth":
            self.head = RandomSynthesizerHead(config)
        elif self.attention_type in ["fcrand_synth", "default_fcrand"]:
            self.head = FactorizedRandomSynthesizerHead(config)

        # Add learnable weight if mixture synthesizer is used
        if self.attention_type == "default_fcrand":
            self.mixture_weight = nn.Parameter(torch.empty(1, self.num_attention_heads, 1, 1, 2), requires_grad=True)
            nn.init.xavier_uniform_(self.mixture_weight)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
    
        if self.add_alibi:
            self.alibi = LinearBiases(config)
        else:
            self.alibi = None
            
        if self.add_urpe:
            self.urpe = URPE(config)
        else:
            self.urpe = None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.attention_type == "default":
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # elif self.attention_type == "dense_synth1" or self.attention_type == "dense_synth2":
        elif self.attention_type in ["dense_synth1", "dense_synth2"]:
            scores_shape =  (hidden_states.size()[0], self.num_attention_heads, self.max_length, self.max_length)
            attention_scores = torch.empty(scores_shape, device=hidden_states.device)
            for i, head_module in enumerate(self.head):
                attention_scores[:, i] = head_module(hidden_states)
        elif self.attention_type in ["rand_synth", "fcrand_synth"]:
            attention_scores = self.head().unsqueeze(0).repeat(hidden_states.size()[0], 1, 1, 1).to(hidden_states.device)
        elif self.attention_type == "default_fcrand":
            # calculate default attention scores
            attention_scores1 = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # build factorized random synthesizer attention scores
            scores_shape =  (self.num_attention_heads, self.max_length, self.max_length)
            attention_scores2 = torch.empty(scores_shape, device=hidden_states.device)
            for i, head_module in enumerate(self.head):
                attention_scores2[i] = head_module(hidden_states)
            attention_scores2 = attention_scores2.unsqueeze(0).repeat(hidden_states.size()[0], 1, 1, 1)

            # combine both attention scores
            mixture_weight = torch.nn.Softmax(dim=-1)(self.mixture_weight)
            attention_scores = mixture_weight[..., 0] * attention_scores1 + mixture_weight[..., 1] * attention_scores2

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        
        if self.add_alibi:
            attention_scores = self.alibi(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if self.add_alibi and self.alibi_type == "nosym_mask":
            attention_scores += self.alibi.nonsym_mask.repeat(attention_scores.size()[0], 1, 1, 1).to(attention_scores.device)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        if self.add_urpe:
            attention_probs = self.urpe(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class EarlyStopping: #https://github.com/Bjarten/early-stopping-pytorch/
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss
 
def mask_all(data, mask_token, mask_proportion=.15):
    seq_len = data['inputs_embeds'].shape[1]
    for i in range(data['inputs_embeds'].shape[0]):
        muted_idx = torch.randperm(seq_len)[:int(np.floor(seq_len * mask_proportion))]
        for j in muted_idx:
            prob = torch.rand(1)
            if prob < 0.8:
                data['inputs_embeds'][i, j, :] = mask_token
            elif prob >= 0.8 and prob < 0.9:
                switch_idx = torch.where(torch.arange(seq_len) != j)[0][torch.randint(high=seq_len-1, size=(1,))]
                data['inputs_embeds'][i, j, :] = data['inputs_embeds'][i, switch_idx, :].squeeze(-1)
            else:
                data['inputs_embeds'][i, j, :] = data['inputs_embeds'][i, j, :]

        data['mask_label'][i, muted_idx] = 1
            
    return data

def generate_queries(bert_data, mask_copy=2, n_shift=3, min_shift_mag=0 , max_shift_mag=25):
    
    train_mlm, test_mlm = {}, {}

    split_idx = int(0.2*bert_data.shape[0])
    set_seed(12315019)
    randn_idx = random.sample(range(0,bert_data.shape[0]),bert_data.shape[0])
    
    train_idx = randn_idx[split_idx:]
    test_idx = randn_idx[:split_idx]

    train_mlm['inputs_embeds'] = bert_data[train_idx].detach().clone()
    train_mlm['labels'] =  bert_data[train_idx].detach().clone()
    train_mlm['mask_label'] = torch.zeros_like(bert_data[train_idx].detach().clone())
    train_mlm['index'] = torch.arange(bert_data[train_idx].shape[0])

    test_mlm['inputs_embeds'] = bert_data[test_idx].detach().clone()
    test_mlm['labels'] =  bert_data[test_idx].detach().clone()
    test_mlm['mask_label'] = torch.zeros_like(bert_data[test_idx].detach().clone())
    test_mlm['index'] = torch.arange(bert_data[test_idx].shape[0])

    filler = torch.mean(train_mlm['inputs_embeds'])
    
    for data in train_mlm, test_mlm:
        data_len = data['inputs_embeds'].shape[0]
        for n in range(n_shift):
            data2 = {key: value[:data_len] for key, value in data.items()}
            for i in range(data_len):
                while True:
                    shift_mag = int(torch.randint(low=min_shift_mag-1, high=max_shift_mag+1, size=(1, )))
                    if shift_mag != 0:
                        break
                data2['inputs_embeds'][i] = torch.roll(data2['inputs_embeds'][i], shift_mag, -1)
                data2['labels'][i] = torch.roll(data2['labels'][i], shift_mag, -1)
                if shift_mag > 0:
                    data2['inputs_embeds'][i, :, :shift_mag] = filler
                    data2['labels'][i, :, :shift_mag] = filler
                elif shift_mag < 0:
                    data2['inputs_embeds'][i, :, data2['inputs_embeds'].shape[-1]+shift_mag:] = filler
                    data2['labels'][i, :, data2['inputs_embeds'].shape[-1]+shift_mag:] = filler

            for key in data.keys():
                data[key] = torch.cat((data[key], data2[key]), 0)

    # Different masking copies
    for key in train_mlm.keys():
        train_mlm[key] = train_mlm[key].repeat(mask_copy, 1, 1)
    train_mlm['index'] = torch.arange(train_mlm['inputs_embeds'].shape[0])
        
    for key in test_mlm.keys():
        test_mlm[key] = test_mlm[key].repeat(mask_copy, 1, 1)
    test_mlm['index'] = torch.arange(test_mlm['inputs_embeds'].shape[0])

    # Apply masking
    set_seed(12315019)
    mask_token = torch.randn(1, 1, train_mlm['inputs_embeds'].shape[-1])
    mask_token = -1 + (2 * (mask_token - torch.min(mask_token)) / (torch.max(mask_token) - torch.min(mask_token)))

    for data_mlm in train_mlm, test_mlm:
        data_mlm = mask_all(data_mlm, mask_token, mask_proportion=.125)
        
    # Polarity reversal
    for d in train_mlm, test_mlm:
        augmented = d.copy()
        augmented['inputs_embeds'] = augmented['inputs_embeds'] *1
        augmented['labels'] = augmented['labels'] *1
        for key in augmented.keys():
            d[key] = torch.cat((d[key], augmented[key]), 0)
        
    return SeismicDataset(train_mlm), SeismicDataset(test_mlm)