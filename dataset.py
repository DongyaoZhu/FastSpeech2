import os
import numpy as np
from multiprocessing import cpu_count
import mindspore as ms

from ljspeech import LJSpeechTTS
from text import text_to_sequence
from utils import get_sinusoid_encoding_table
from tts_dataset import DistributedSampler, create_base_dataset


WAV_DIR = 'wavs'
MEL_DIR = '../../ptFastSpeech2/preprocessed_data/LJSpeech_paper/mel'
ENERGY_DIR = '../../ptFastSpeech2/preprocessed_data/LJSpeech_paper/energy'
PITCH_DIR = '../../ptFastSpeech2/preprocessed_data/LJSpeech_paper/pitch'
DURATION_DIR = '../../ptFastSpeech2/preprocessed_data/LJSpeech_paper/duration'

WAV_POSTFIX = '.wav'
MEL_POSTFIX = 'LJSpeech-mel-' # '_m_fs2.npy'
ENERGY_POSTFIX = 'LJSpeech-energy-' # '_e_fs2.npy'
PITCH_POSTFIX = 'LJSpeech-pitch-' # '_p_fs2.npy'
DURATION_POSTFIX = 'LJSpeech-duration-' # '_d_fs2.npy'

len_max_seq = 1000
d_word_vec = 256
positional_embeddings = get_sinusoid_encoding_table(len_max_seq + 1, d_word_vec, padding_idx=None)

from hparams import hps

def read_all_phonemes(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = {}
        for line in f.readlines():
            n, s, t, r = line.strip("\n").split("|")
            data[n] = {
                'name': n,
                'text': t,
                'raw_text': r
            }
    return data


def create_dataset(data_path, manifest_path, batch_size, is_train=True, rank=0, group_size=1):
    ds = LJSpeechTTS(
        data_path=data_path,
        manifest_path=manifest_path,
        is_train=is_train,
    )
    ds = create_base_dataset(ds, rank=rank, group_size=group_size)
    all_phonemes = read_all_phonemes('../ptFastSpeech2/preprocessed_data/LJSpeech_paper/train.txt')
    all_phonemes.update(read_all_phonemes('../ptFastSpeech2/preprocessed_data/LJSpeech_paper/val.txt'))
    def replace(src, patterns, options):
        for p, o in zip(patterns, options):
            src = src.replace(p, o)
        return src
    def read_feat(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        base = filename[filename.rfind('/')+1: ].replace(WAV_POSTFIX, '')

        phonemes = np.array(text_to_sequence(all_phonemes[base]['text'], ["english_cleaners"]))
        # mel = np.load(filename.replace(WAV_DIR, MEL_DIR).replace(base, MEL_POSTFIX + base).replace(WAV_POSTFIX, '.npy'))
        mels = np.load(replace(filename, [WAV_DIR, base, WAV_POSTFIX], [MEL_DIR, MEL_POSTFIX + base, '.npy']))
        pitch = np.load(replace(filename, [WAV_DIR, base, WAV_POSTFIX], [PITCH_DIR, PITCH_POSTFIX + base, '.npy']))
        energy = np.load(replace(filename, [WAV_DIR, base, WAV_POSTFIX], [ENERGY_DIR, ENERGY_POSTFIX + base, '.npy']))
        duration = np.load(replace(filename, [WAV_DIR, base, WAV_POSTFIX], [DURATION_DIR, DURATION_POSTFIX + base, '.npy']))
        return phonemes, mels, pitch, energy, duration

    output_columns = ['phonemes', 'mels', 'pitch', 'energy', 'duration']
    ds = ds.map(
        input_columns=['audio'],
        output_columns=output_columns,
        column_order=output_columns,
        operations=[read_feat],
        num_parallel_workers=cpu_count(),
    )

    input_columns = output_columns
    output_columns = [
        'speakers',
        'texts',
        'src_lens',
        'max_src_len',
        'positions_encoder',
        'positions_decoder',
        'mels',
        'mel_lens',
        'max_mel_len',
        'p_targets',
        'e_targets',
        'd_targets',
    ]
    def pad_to_max(xs):
        B = len(xs)
        T = max(x.shape[0] for x in xs)
        shape = [B, T] + list(xs[0].shape[1:])
        ys = np.zeros(shape)
        lengths = np.zeros(B, np.int32)
        for i, x in enumerate(xs):
            ys[i, : x.shape[0]] = x
            lengths[i] = x.shape[0]
        return ys, lengths, np.array(T, np.int32)

    def batch_collate(phonemes, mels, pitch, energy, duration, unused_batch_info=None):
        phonemes, src_lens, max_src_len = pad_to_max(phonemes)
        mels, mel_lens, max_mel_len = pad_to_max(mels)
        pitch, _, _ = pad_to_max(pitch)
        energy, _, _ = pad_to_max(energy)
        duration, _, _ = pad_to_max(duration)
        speakers = np.zeros(len(phonemes), np.float32)
        positions_encoder = positional_embeddings[None, : max_src_len].repeat(len(phonemes), 0)
        max_duration = duration.sum(-1).max().astype(np.int32)
        positions_decoder = positional_embeddings[None, : max_duration].repeat(len(phonemes), 0)
        # print('dur:', max_duration, 'mel:', mels.shape, 'maxmel:', max_mel_len)
        return (
            speakers,
            phonemes,
            src_lens,
            max_src_len,
            positions_encoder,
            positions_decoder,
            mels,
            mel_lens,
            max_mel_len,
            pitch,
            energy,
            duration.astype(np.int32),
        )
    ds = ds.batch(
        batch_size, 
        per_batch_map=batch_collate,
        input_columns=input_columns,
        output_columns=output_columns,
        column_order=output_columns,
        drop_remainder=True,
        python_multiprocessing=False,
        num_parallel_workers=8
    )

    return ds

if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='CPU')#, device_id=6)
    ds = create_dataset(hps.data_path, hps.manifest_path, hps.batch_size)
    it = ds.create_dict_iterator()
    for nb, d in enumerate(it):
        print('nb:', nb)
        for k, v in d.items():
            print('k:', k, 'v:', v.shape, 't:', v.dtype)
        break