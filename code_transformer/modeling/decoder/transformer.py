import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, MultiheadAttention, LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.modules.transformer import TransformerDecoderLayer

from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.modeling.code_transformer.code_transformer import _get_activation_fn
from code_transformer.modeling.code_transformer.lm import CodeTransformerOutput
from code_transformer.modeling.decoder.pointer import PointerNetwork, Rank1PointerNetwork
from code_transformer.utils.data import batch_index_select

Tensor = torch.Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, position):
        x = x + self.pe[position, :]
        return self.dropout(x)


class TransformerLMDecoder(nn.Module):

    def __init__(self, config: TransformerLMDecoderConfig):
        super(TransformerLMDecoder, self).__init__()

        self.lm_encoder = config.lm_encoder

        self.sos_id = config.sos_id
        self.unk_id = config.unk_id
        self.output_subtokens_per_token = config.output_subtokens_per_token
        self.use_separate_vocab = config.target_vocab_size is not None
        # If target_vocab_size is set, a separate vocabulary for input and output tokens is assumed
        self.vocab_size = config.target_vocab_size if self.use_separate_vocab else config.lm_encoder.vocab_size
        self.d_model = config.lm_encoder.d_model
        n_heads = config.decoder_nhead

        self.n_layers = config.n_layers
        self.use_teacher_forcing = config.use_teacher_forcing

        self.output_nonlinearity = None
        if config.output_nonlinearity is not None:
            self.output_nonlinearity = _get_activation_fn(config.output_nonlinearity)

        self.loss_fct = config.loss_fct

        self.use_pointer_network = config.use_pointer_network
        self.use_pointer_query_linear = config.use_pointer_query_linear
        self.use_pointer_query_self_attention = config.use_pointer_query_self_attention
        self.concat_query_and_pointer = config.concat_query_and_pointer
        self.attend_cls_token = config.attend_cls_token

        self.rank1 = config.rank1 
        self.pointer_rank1 = config.pointer_rank1 

        if self.rank1:
            decoder_layer = Rank1TransformerDecoderLayer(
                self.d_model, config.decoder_nhead, config.decoder_dim_feedforward,
                config.decoder_dropout, config.decoder_activation)
            self.transformer_decoder = Rank1TransformerDecoder(decoder_layer, self.n_layers)
        else:
            decoder_layer = TransformerDecoderLayer(self.d_model, config.decoder_nhead, config.decoder_dim_feedforward,
                                                    config.decoder_dropout, config.decoder_activation)
            self.transformer_decoder = TransformerDecoder(decoder_layer, self.n_layers)

        self.positional_encoding = PositionalEncoding(self.d_model, config.decoder_dropout)

        if self.use_pointer_network:
            if self.pointer_rank1:
                self.pointer_network = Rank1PointerNetwork(self.d_model, self.lm_encoder.subtokens_per_token,
                                                          config.pointer_attention_type,
                                                          n_heads)
            else:
                self.pointer_network = PointerNetwork(self.d_model, self.lm_encoder.subtokens_per_token,
                                                    config.pointer_attention_type,
                                                    n_heads)

            if self.concat_query_and_pointer:
                self.pointer_query_linear = nn.Linear(self.d_model * 2, self.d_model)
                self.pointer_query_nonlinearity = _get_activation_fn('tanh')

            if self.use_pointer_query_self_attention:
                self.pointer_query_self_attention = MultiheadAttention(self.d_model, n_heads,
                                                                       dropout=config.decoder_dropout)
                self.pointer_query_norm = LayerNorm(self.d_model)

        if self.use_separate_vocab:
            self.target_token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for name, p in self.named_parameters():
            if p.dim() > 1:
                if not 'alpha' in name and not 'gamma' in name:
                    xavier_uniform_(p)

    def forward(self,
                labels=None,
                extended_vocabulary_ids=None,
                pointer_pad_mask: Optional[torch.Tensor] = None,
                **model_input) -> CodeTransformerOutput:
        """
        :param labels:
        :param extended_vocabulary_ids: torch.Tensor [B, subtoken_seq_len]
            Defines a sequence of subtokens for every sample. Can be seen as a flattened version of the tokens input
            with UNKNOWN_TOKENs replaced by incremental artificial vocabulary IDs that are only valid for this sample.
            Needed for the pointer mechanism
        :param pointer_pad_mask: torch.Tensor [B, S, num_subtokens]
            A mask that specifies padding down to subtoken level. Needed for the pointer mechanism as we need to point
            to distinct subtokens. 1 indicates that the respective subtoken is NOT a PAD token
        :param model_input:
            additional inputs that are passed on to the encoder.
            The TransformerDecoder expects that model_input has at least the following entries:
                - pad_mask: [B, S], where 1 indicates that the position is a PAD token
                - attention_mask: [B, S, S], where a 1 at [:,i,j] indicates that position i may not attend position j
        :return:
        """

        device = next(self.parameters()).device
        pointer_gates = None
        pointer_attentions = None
        pointer_attention_distributions = None

        transformer_output = self.lm_encoder.forward(need_all_embeddings=True, **model_input)

        B = transformer_output[0].shape[0]
        S = transformer_output.all_emb[-1][0].shape[0]
        V = self.vocab_size
        D = self.d_model

        # all_emb[-1][0] takes the content stream of the last layer, all_emb[-1][1] would take query stream
        content_stream_emb = transformer_output.all_emb[-1][0].transpose(0, 1)  # [B, S, D]
        query_stream_emb = transformer_output.all_emb[-1][1]  # [n_predict, B, D]
        n_predict = transformer_output[0].shape[1]
        if n_predict > 1:
            content_stream_emb = content_stream_emb \
                .unsqueeze(1) \
                .repeat((1, n_predict, 1, 1)) \
                .reshape(B * n_predict, S, D)
            labels = labels.reshape(B * n_predict, 1, -1)
            B = B * n_predict

        # Initially start decoding with a sequence containing only one <s> token per sample
        # Input tokens should have B x T x D
        # every batch decoding starts with the same initial input
        initial_input = torch.tensor([[self.sos_id]], device=device)
        token_embedding = self.target_token_embedding if self.use_separate_vocab else self.lm_encoder.token_embedding
        decoder_input = token_embedding(initial_input).expand((B, -1, -1))
        decoder_input = self.positional_encoding.forward(decoder_input, 0)

        if self.use_pointer_network:
            pointer_input_subtokens = content_stream_emb
            if n_predict > 1:
                pointer_pad_mask = pointer_pad_mask.unsqueeze(1) \
                    .repeat(1, n_predict, 1, 1) \
                    .reshape(B, S, -1)
                extended_vocabulary_ids = extended_vocabulary_ids.unsqueeze(1) \
                    .repeat(1, n_predict, 1) \
                    .reshape(B, -1)

            if self.pointer_rank1:
                self.pointer_network.init_batch(pointer_input_subtokens, pointer_pad_mask, extended_vocabulary_ids,
                                                self.vocab_size, languages=model_input['languages'])
            else:
                self.pointer_network.init_batch(pointer_input_subtokens, pointer_pad_mask, extended_vocabulary_ids,
                                                self.vocab_size)

            logits = torch.zeros((self.output_subtokens_per_token, B, self.pointer_network.len_extended_vocab),
                                 device=device)
            pointer_gates = torch.zeros((self.output_subtokens_per_token, B))
            pointer_attentions = torch.zeros(
                (self.output_subtokens_per_token, B, self.pointer_network.len_extended_vocab))
            pointer_attention_distributions = torch.zeros(
                (self.output_subtokens_per_token, B, extended_vocabulary_ids.shape[1])
            )
        else:
            logits = torch.zeros((self.output_subtokens_per_token, B, V), device=device)

        # pad_mask has 1s for all regular (non-pad) tokens
        # attention_mask has 1s for all illegal tokens that may not be attended (such as function name and CLS token)
        pad_mask = model_input['pad_mask'].bool()
        if n_predict > 1:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, n_predict, 1).reshape(B, -1)
            attention_mask = model_input['attention_mask']
            label_idx = torch.stack([torch.where(tm == 1)[0] for tm in model_input['target_mapping'].sum(dim=1)])
            attention_mask = batch_index_select(attention_mask, dim=1, index=label_idx)
            attention_mask = attention_mask.reshape(B, -1)
        else:
            attention_mask = model_input['attention_mask']
            attention_mask = torch.stack(
                [attention_mask[i][torch.where(model_input['pad_mask'][i] == 0)[0]].sum(dim=0) for i in range(B)])
        attention_mask = attention_mask > 0
        if self.attend_cls_token:
            attention_mask[:, 0] = False  # CLS token may be attended

        for idx in range(self.output_subtokens_per_token):

            if self.use_pointer_network:
                if self.concat_query_and_pointer:
                    pointer_query = decoder_input.select(1, -1)
                    pointer_query = torch.cat([pointer_query, query_stream_emb.reshape(B, D)], dim=1)

                    pointer_query = self.pointer_query_linear(pointer_query)
                    pointer_query = self.pointer_query_nonlinearity(pointer_query)
                else:
                    pointer_query = decoder_input.select(1, -1)

                if self.use_pointer_query_self_attention:
                    pointer_query = \
                        self.pointer_query_self_attention(pointer_query.unsqueeze(0), decoder_input.transpose(0, 1),
                                                          decoder_input.transpose(0, 1))[0]
                    pointer_query = self.pointer_query_norm(pointer_query)

                    pointer_query = pointer_query.squeeze(0)

                if self.pointer_rank1:
                    self.pointer_network.calculate_pointer_attention(pointer_query, languages=model_input['languages'])
                else:
                    self.pointer_network.calculate_pointer_attention(pointer_query)

            if self.rank1:
                decoder_output = self.transformer_decoder.forward(decoder_input.transpose(0, 1),
                                                                content_stream_emb.transpose(0, 1),
                                                                memory_key_padding_mask=pad_mask | attention_mask,
                                                                languages=model_input['languages'])
            else:
                decoder_output = self.transformer_decoder.forward(decoder_input.transpose(0, 1),
                                                                content_stream_emb.transpose(0, 1),
                                                                memory_key_padding_mask=pad_mask | attention_mask)
            if self.output_nonlinearity is not None:
                decoder_output = self.output_nonlinearity(decoder_output)

            # B x V
            subtoken_logits = decoder_output.select(0, -1) @ token_embedding.weight.T

            if self.use_pointer_network:
                subtoken_logits = self.pointer_network.combine_probabilites(subtoken_logits)
                pointer_gates[idx] = self.pointer_network.pointer_gate.squeeze(-1)
                pointer_attentions[idx] = self.pointer_network.pointer_attention
                pointer_attention_distributions[idx] = self.pointer_network.pointer_attention_distribution

            logits[idx] = subtoken_logits

            # Calculate next decoder_input
            if self.use_teacher_forcing and self.training:
                # Use previous label as next input
                next_input = labels[:, :, idx]  # B x 1
            else:
                next_input = subtoken_logits.argmax(-1).detach().unsqueeze(1)  # B x 1

            if self.use_pointer_network:
                next_input = self.pointer_network.get_next_input(next_input, self.unk_id)

            next_input_embedding = token_embedding(next_input)
            next_input_embedding = self.positional_encoding.forward(next_input_embedding, idx + 1)
            next_input = torch.cat([decoder_input, next_input_embedding], 1)
            decoder_input = next_input

        loss = self.loss_fct(logits.transpose(0, 1).reshape(-1, logits.size(-1)), labels.view(-1))

        logits = logits.transpose(0, 1).unsqueeze(1)  # B x 1 x output_subtokens x V
        logits = logits.reshape(B // n_predict, n_predict, logits.shape[2], logits.shape[3])
        outputs = CodeTransformerOutput(loss=loss,
                                        logits=logits,
                                        attentions=transformer_output.attentions,
                                        pointer_gates=pointer_gates,
                                        pointer_attentions=pointer_attentions,
                                        pointer_attention_distributions=pointer_attention_distributions)

        return outputs


class Rank1TransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-05, batch_first=False, device=None, dtype=None):
        super(Rank1TransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward,
                                                           dropout, activation)
        self.self_attn = Rank1MultiheadAttention(d_model, nhead, dropout=dropout, share=True)
        self.multihead_attn = Rank1MultiheadAttention(d_model, nhead, dropout=dropout)

        num_languages = 4
        self.alpha = nn.Parameter(torch.FloatTensor(num_languages, d_model))
        self.gamma = nn.Parameter(torch.FloatTensor(num_languages, 3 * d_model))

        self.linear1_alpha = nn.Parameter(torch.FloatTensor(num_languages, d_model))
        self.linear1_gamma = nn.Parameter(torch.FloatTensor(num_languages, dim_feedforward))
        self.linear2_alpha = nn.Parameter(torch.FloatTensor(num_languages, dim_feedforward))
        self.linear2_gamma = nn.Parameter(torch.FloatTensor(num_languages, d_model))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None, 
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                languages: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              languages=languages)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   languages=languages)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        alpha = torch.index_select(self.linear1_alpha, 0, languages).unsqueeze(0)
        gamma = torch.index_select(self.linear1_gamma, 0, languages).unsqueeze(0)
        perturb_tgt =  tgt * alpha
        linear1_output = self.linear1(perturb_tgt) * gamma

        linear2_input = self.dropout(self.activation(linear1_output))
        alpha = torch.index_select(self.linear2_alpha, 0, languages).unsqueeze(0)
        gamma = torch.index_select(self.linear2_gamma, 0, languages).unsqueeze(0)
        perturb_linear2_input = linear2_input * alpha
        tgt2 = self.linear2(perturb_linear2_input) * gamma
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _reset_parameters(self):
        # TODO: reset rank1 parameters.
        nn.init.constant_(self.alpha, 1.)
        nn.init.constant_(self.gamma, 1.)
        nn.init.constant_(self.linear1_alpha, 1.)
        nn.init.constant_(self.linear1_gamma, 1.)
        nn.init.constant_(self.linear2_alpha, 1.)
        nn.init.constant_(self.linear2_gamma, 1.)

        #nn.init.normal_(self.alpha, mean=1., std=0.5)
        #nn.init.normal_(self.gamma, mean=1., std=0.5) 
        #nn.init.normal_(self.linear1_alpha, mean=1., std=0.5)
        #nn.init.normal_(self.linear1_gamma, mean=1., std=0.5)
        #nn.init.normal_(self.linear2_alpha, mean=1., std=0.5)
        #nn.init.normal_(self.linear2_gamma, mean=1., std=0.5)
        self.self_attn._reset_parameters()
        self.multihead_attn._reset_parameters()


class Rank1TransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers):
        super(Rank1TransformerDecoder, self).__init__(decoder_layer, num_layers)
        self._reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                languages:Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         languages=languages)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for l in self.layers:
            l._reset_parameters()


class Rank1MultiheadAttention(MultiheadAttention):
    def __init__(self, d_model, nhead, dropout=0.1, share=False):
        super(Rank1MultiheadAttention, self).__init__(d_model, nhead, dropout)

        # TODO: Strange here
        self.batch_first = False

        num_languages = 4
        self.share = share
        if share:
            self.alpha = nn.Parameter(torch.FloatTensor(num_languages, d_model))
        else:
            self.alpha = nn.Parameter(torch.FloatTensor(num_languages, 2 * d_model))
        self.gamma = nn.Parameter(torch.FloatTensor(num_languages, 3 * d_model))

        self.linear_alpha = nn.Parameter(torch.FloatTensor(num_languages, d_model)) 
        self.linear_gamma = nn.Parameter(torch.FloatTensor(num_languages, d_model)) 
        self.d_model = d_model

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
                languages: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        alpha = torch.index_select(self.alpha, 0, languages).unsqueeze(0)
        gamma = torch.index_select(self.gamma, 0, languages).unsqueeze(0)
        linear_alpha = torch.index_select(self.linear_alpha, 0, languages).unsqueeze(0)
        linear_gamma = torch.index_select(self.linear_gamma, 0, languages).unsqueeze(0)

        if self.share:
            query = query * alpha
            key = key * alpha
            value = value * alpha
        else:
            query = query * alpha[:, :, :self.d_model]
            key = key * alpha[:, :, self.d_model:2*self.d_model]
            value = value * alpha[:, :, self.d_model:2*self.d_model]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = rank1_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                gamma, linear_alpha, linear_gamma,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = rank1_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                gamma, linear_alpha, linear_gamma,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
    
    def _reset_parameters(self):
        super()._reset_parameters()
        if hasattr(self, 'alpha'):
            nn.init.constant_(self.alpha, 1.)
            nn.init.constant_(self.gamma, 1.)
            nn.init.constant_(self.linear_alpha, 1.)
            nn.init.constant_(self.linear_gamma, 1.)

            #nn.init.normal_(self.alpha, mean=1., std=0.5)
            #nn.init.normal_(self.gamma, mean=1., std=0.5) 
            #nn.init.normal_(self.linear_alpha, mean=1., std=0.5)
            #nn.init.normal_(self.linear_gamma, mean=1., std=0.5)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def rank1_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    gamma: Tensor, 
    linear_alpha: Tensor, 
    linear_gamma: Tensor, 
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    q = q * gamma[:, :, :embed_dim_to_check]
    k = k * gamma[:, :, embed_dim_to_check:2*embed_dim_to_check]
    v = v * gamma[:, :, 2*embed_dim_to_check:3*embed_dim_to_check]
    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        import warnings
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    perturb_attn_output = attn_output * linear_alpha
    attn_output = F.linear(perturb_attn_output, out_proj_weight, out_proj_bias)
    attn_output *= linear_gamma 

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None