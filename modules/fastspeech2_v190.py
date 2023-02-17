import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.amp import DynamicLossScaler

import os
import json

import sys
sys.path.append('..')
from modules.transformer.models import Encoder, Decoder
from modules.variance_adapter import VarianceAdaptor
from utils import get_mask_from_lengths


class FastSpeech2(nn.Cell):
    def __init__(self, preprocess_config, model_config):
        super().__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Dense(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        yh = self.variance_adaptor(
            x=output,
            src_mask=src_masks,
            mel_mask=mel_masks,
            max_len=int(max_mel_len.asnumpy()),
            pitch_target=p_targets,
            energy_target=e_targets,
            duration_target=d_targets,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )
        output, mel_masks = self.decoder(yh['output'], yh['mel_masks'])
        output = self.mel_linear(output)
        yh.update({
            'mel_predictions': output,
            'mel_masks': mel_masks,
            'src_masks': src_masks,
            'src_lens': src_lens
        })
        return yh

    def construct(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FastSpeech2WithLoss(FastSpeech2):
    def __init__(self, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.scale = DynamicLossScaler(1024, 2, 1)

    def construct(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets,
        e_targets,
        d_targets,
    ):
        yh = self.forward(
            speakers=speakers,
            texts=texts,
            src_lens=src_lens,
            max_src_len=max_src_len,
            mel_lens=mel_lens,
            max_mel_len=max_mel_len,
            p_targets=p_targets,
            e_targets=e_targets,
            d_targets=d_targets,
        )
        yh.update({
            'mel_targets': mels,
            'pitch_targets': p_targets,
            'energy_targets': e_targets,
            'duration_targets': d_targets,
        })
        # return self.loss_fn(yh)
        return self.scale.scale(self.loss_fn(yh))


if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    # ms.context.set_context(mode=ms.context.PYNATIVE_MODE)

    b, c, t = 2, 128, 7200
    x = np.random.random([b, t]).astype(np.float32)
    s = np.random.random([b, ]).astype(np.float32)
    n = np.random.random([b, t]).astype(np.float32)
    c = np.random.random([b, c, t // 300]).astype(np.float32)

    net = FastSpeech2WithLoss(1)
    y = net(ms.Tensor(x), ms.Tensor(s), ms.Tensor(n), ms.Tensor(c))
    print('y:', y.shape)