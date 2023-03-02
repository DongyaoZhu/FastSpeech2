# Given the path to ljspeech/wavs,
# this script converts wav files to .npy features used for training.

import sys, os
import numpy as np
import pyworld as pw
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from mindspore.dataset.audio import Spectrogram, MelScale
import mindspore as ms
# from mindaudio.data.io import read
# import librosa

sys.path.append('.')
from text import text_to_sequence
from phonemes import get_alignment
from hparams import hps
from ljspeech import LJSpeechTTS
from dataset import (
    create_base_dataset,
    feature_columns,
    all_dirs,
    all_postfix,
)

def read_wav(filename):
    filename = str(filename).replace('b\'', '').replace('\'', '')
    audio = np.load(filename.replace('.wav', '_wav.npy'))
    # audio, sr = librosa.load(filename, sr=hps.sample_rate)
    # audio = signal.lfilter([1], [1, -hps.preemph_coef], audio).astype(np.float32)
    # audio = np.clip(audio, -1., 1.)
    return audio, audio, filename

def read_wav_ms(filename):
    filename = str(filename).replace('b\'', '').replace('\'', '')
    from mindaudio.data.io import read
    audio, _ = read(filename)
    signed_int16_max = 2**15
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / signed_int16_max
    audio = audio / np.max(np.abs(audio))
    return audio, audio, filename

stft_fn = Spectrogram(
    n_fft=hps.n_fft,
    win_length=hps.hop_samples * 4,
    hop_length=hps.hop_samples,
    power=1.,
    center=True,
)

mel_fn = MelScale(
    n_mels=hps.n_mels,
    sample_rate=hps.sample_rate,
    f_min=20., 
    f_max=hps.sample_rate / 2.0,
    n_stft=hps.n_fft // 2 + 1,
)

def _normalize(S):
    S = 20 * np.log10(np.clip(S, 1e-5, None)) - 20
    S = np.clip((S + 100) / 100, 0.0, 1.0)
    return S

# process text: file -> phoneme
def get_fs2_features(audio, text):
    text = str(text).replace('b\'', '').replace('\'', '')
    base = text[text.rfind('/')+1: ].replace('.txt', '')
    tg_path = os.path.join(hps.data_path, 'TextGrid/LJSpeech', f'{base}.TextGrid')
    phoneme, duration, start, end = get_alignment(tg_path, hps.sample_rate, hps.hop_samples)
    with open(text, 'r') as f:
        raw_text = f.readline().strip('\n')
    phoneme = "{" + " ".join(phoneme) + "}"
    base = "|".join([base, 'ljspeech', phoneme, raw_text])
    phoneme = np.array(text_to_sequence(phoneme, ["english_cleaners"]))
    # '''
    wav, _, filename = read_wav(audio)
    wav = wav[int(hps.sample_rate * start) : int(hps.sample_rate * end)]

    pitch, t = pw.dio(
        wav.astype(np.float64),
        hps.sample_rate,
        frame_period=hps.hop_samples / hps.sample_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, hps.sample_rate)[: sum(duration)]

    S = stft_fn(wav)
    energy = np.linalg.norm(S, axis=0)[: sum(duration)]
    mel = _normalize(mel_fn(S)[:, : sum(duration)])
    # ['phoneme', 'wav', 'mel', 'energy', 'pitch', 'duration', 'base']
    # '''
    # wav = mel = energy = pitch = 0
    return phoneme, wav, mel, energy, pitch, duration, base

def create_prep_dataset(data_path, manifest_path, is_train):
    ds = LJSpeechTTS(
        data_path=data_path,
        manifest_path=manifest_path,
        is_train=is_train
    )
    ds = create_base_dataset(ds, rank=0, group_size=1)

    # process audio: file -> wav -> feature
    ds = ds.map(
        input_columns=['audio', 'text'],
        output_columns=feature_columns + ['base'],
        column_order=feature_columns + ['base'],
        operations=get_fs2_features,
        num_parallel_workers=cpu_count(),
    )

    return ds


def preprocess_ljspeech(data_path, manifest_path, is_train):
    ds = create_prep_dataset(data_path, manifest_path, is_train)
    it = ds.create_dict_iterator()

    results = []
    pool = Pool(processes=cpu_count())

    for k in feature_columns:
        os.makedirs(os.path.join(data_path, all_dirs[k]), exist_ok=True)

    for x in tqdm(it, total=ds.get_dataset_size()):
        base = str(x['base']).split('|', 1)[0]
        with open(os.path.join(data_path, all_dirs['phoneme'], base + '_phoneme.txt'), 'w') as writer:
            writer.write(str(x['base']) + '\n')
        for k in feature_columns:
            # if k not in {'phoneme'}:
                # continue
            np.save(os.path.join(data_path, all_dirs[k], base + all_postfix[k]), x[k].asnumpy())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', '-d', type=str, default="CPU", choices=("GPU", "CPU", 'Ascend'))
    parser.add_argument('--device_id', '-i', type=int, default=0)
    args = parser.parse_args()
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)
    # preprocess_ljspeech(
    #     data_path=hps.data_path,
    #     manifest_path=hps.manifest_path,
    #     is_train=False
    # )
    preprocess_ljspeech(
        data_path=hps.data_path,
        manifest_path=hps.manifest_path,
        is_train=True
    )
