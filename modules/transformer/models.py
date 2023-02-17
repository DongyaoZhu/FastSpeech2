import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from modules.transformer import constants
from modules.transformer.layers import FFTBlock
from text.symbols import all_symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return Tensor(sinusoid_table, dtype=mstype.float32)


class Encoder(nn.Cell):
    def __init__(self, config):
        super().__init__()

        n_src_vocab = len(all_symbols) + 1
        len_max_seq = config["max_seq_len"]
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = config["transformer"]["encoder_hidden"] // config["transformer"]["encoder_head"]
        d_v = config["transformer"]["encoder_hidden"] // config["transformer"]["encoder_head"]
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        n_position = len_max_seq + 1
        pretrained_embs = get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0)

        self.position_enc = nn.Embedding(
            n_position,
            d_word_vec,
            embedding_table=pretrained_embs,
            padding_idx=0,
        )

        self.src_word_emb = nn.Embedding(
            n_src_vocab,
            d_word_vec,
            padding_idx=constants.PAD,
        )

        self.layer_stack = nn.CellList(
            [
                FFTBlock(d_model, d_inner, kernel_size, n_head, d_k, d_v, dropout) for _ in range(n_layers)
            ]
        )

        self.equal = ops.Equal()
        self.not_equal = ops.NotEqual()
        self.expand_dims = ops.ExpandDims()
        self.pad = constants.PAD

    def construct(self, src_seq, src_pos):
        """
        Create mask and forward to FFT blocks.

        Args:
            src_seq (Tensor): Tokenized text sequence. Shape (hp.batch_size, hp.character_max_length).
            src_pos (Tensor): Positions of the sequences. Shape (hp.batch_size, hp.character_max_length).

        Returns:
            enc_output (Tensor): Encoder output.
        """
        # Prepare masks
        padding_mask = self.equal(src_seq, self.pad)
        slf_attn_mask = self.expand_dims(padding_mask.astype(mstype.float32), 1)
        slf_attn_mask_bool = slf_attn_mask.astype(mstype.bool_)

        non_pad_mask_bool = self.expand_dims(self.not_equal(src_seq, self.pad), 2)
        non_pad_mask = non_pad_mask_bool.astype(mstype.float32)

        # Forward
        enc_output = self.src_word_emb(src_seq.astype('int32')) + self.position_enc(src_pos.astype('int32'))
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_bool,
            )
        return enc_output


class Decoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        n_position = config["max_seq_len"] + 1
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = config["transformer"]["decoder_hidden"] // config["transformer"]["decoder_head"]
        d_v = config["transformer"]["decoder_hidden"] // config["transformer"]["decoder_head"]
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        len_max_seq = config["max_seq_len"]
        self.max_seq_len = config["max_seq_len"]

        n_position = len_max_seq + 1

        pretrained_embs = get_sinusoid_encoding_table(n_position, d_model, padding_idx=0)

        self.position_enc = ops.stop_gradient(pretrained_embs.expand_dims(0))

        self.layer_stack = nn.CellList(
            [
                FFTBlock(d_model, d_inner, kernel_size, n_head, d_k, d_v, dropout) for _ in range(n_layers)
            ]
        )

        self.pad = constants.PAD
        self.equal = ops.Equal()
        self.not_equal = ops.NotEqual()
        self.expand_dims = ops.ExpandDims()

    def construct(self, enc_seq, mask):
        slf_attn_mask = self.expand_dims(mask.astype(mstype.float32), 1)
        slf_attn_mask_bool = slf_attn_mask.astype(mstype.bool_)

        non_pad_mask = 1. - mask.expand_dims(2)

        # Forward
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        dec_output = enc_seq + self.position_enc[:, :max_len, :]

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_bool)

        return dec_output, slf_attn_mask_bool
