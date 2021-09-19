#!/usr/bin/env python3

# Copyright 2021 Yifan Peng
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Speech Commands Dataset: https://arxiv.org/abs/1804.03209
# Our data preparation is similar to the TensorFlow script:
# https://www.tensorflow.org/datasets/catalog/speech_commands


import os
import os.path
import glob
import argparse
import numpy as np
from scipy.io import wavfile


parser = argparse.ArgumentParser(description="Process speech commands dataset.")
parser.add_argument(
    '--data_path', 
    type=str, 
    default='downloads/speech_commands_v0.02', 
    help='folder containing the original data'
)
parser.add_argument(
    '--test_data_path',
    type=str,
    default='downloads/speech_commands_test_set_v0.02',
    help='folder containing the test set'
)
parser.add_argument(
    '--train_dir',
    type=str,
    default='data/train',
    help='output folder for training data'
)
parser.add_argument(
    '--dev_dir',
    type=str,
    default='data/dev',
    help='output folder for validation data'
)
parser.add_argument(
    '--test_dir',
    type=str,
    default='data/test',
    help='output folder for test data'
)
args = parser.parse_args()


WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
SILENCE = '_silence_'
UNKNOWN = '_unknown_'
LABELS = WORDS + [SILENCE, UNKNOWN]     # 12 labels in the test set
BACKGROUND_NOISE = '_background_noise_'
SAMPLE_RATE = 16000

# Generate test data
with open(
    os.path.join(args.test_dir, 'text'), 'w'
) as text_f, open(
    os.path.join(args.test_dir, 'wav.scp'), 'w'
) as wav_scp_f, open(
    os.path.join(args.test_dir, 'utt2spk'), 'w'
) as utt2spk_f:
    for label in LABELS:
        wav_list = [n for n in os.listdir(os.path.join(args.test_data_path, label)) if n.endswith('.wav')]
        for wav in wav_list:
            uttid = f'{label.strip("_")}_{wav.rstrip(".wav")}'
            text_f.write(uttid + ' ' + label + '\n')
            wav_scp_f.write(uttid + ' ' + os.path.abspath(os.path.join(args.test_data_path, label, wav)) + '\n')
            utt2spk_f.write(uttid + ' ' + uttid + '\n')

# Generate train and dev data
with open(os.path.join(args.data_path, 'validation_list.txt'), 'r') as dev_f:
    dev_file_list = [line.rstrip() for line in dev_f.readlines()]
    # add running_tap into the dev set
    dev_file_list.append(os.path.join(BACKGROUND_NOISE, 'running_tap.wav'))
    dev_file_list = [os.path.abspath(os.path.join(args.data_path, line)) for line in dev_file_list]
with open(os.path.join(args.data_path, 'testing_list.txt'), 'r') as test_f:
    test_file_list = [line.rstrip() for line in test_f.readlines()]
    test_file_list = [os.path.abspath(os.path.join(args.data_path, line)) for line in test_file_list]

full_file_list = [os.path.abspath(p) for p in glob.glob(os.path.join(args.data_path, '*', '*.wav'))]
train_file_list = list(set(full_file_list) - set(dev_file_list) - set(test_file_list))

for name in ['train', 'dev']:
    if name == 'train':
        file_list = train_file_list
        out_dir = args.train_dir
    else:
        file_list = dev_file_list
        out_dir = args.dev_dir
    
    with open(
        os.path.join(out_dir, 'text'), 'w'
    ) as text_f, open(
        os.path.join(out_dir, 'wav.scp'), 'w'
    ) as wav_scp_f, open(
        os.path.join(out_dir, 'utt2spk'), 'w'
    ) as utt2spk_f:
        for wav_abspath in file_list:  # absolute path
            word, wav = wav_abspath.split('/')[-2:]
            if word != BACKGROUND_NOISE:
                if word in WORDS:
                    label = word
                else:
                    label = UNKNOWN
                uttid = f'{word.strip("_")}_{wav.rstrip(".wav")}'
                text_f.write(uttid + ' ' + label + '\n')
                wav_scp_f.write(uttid + ' ' + wav_abspath + '\n')
                utt2spk_f.write(uttid + ' ' + uttid + '\n')
            else:
                processed_dir = os.path.join(args.data_path, BACKGROUND_NOISE, 'processed')
                os.makedirs(
                    processed_dir,
                    exist_ok=True
                )
                label = SILENCE
                # split the original audio to 1-second clips
                try:
                    wav_rate, wav_data = wavfile.read(wav_abspath)  # 1-D array
                except:
                    print(wav)
                else:
                    assert wav_rate == SAMPLE_RATE
                    for start in range(0, wav_data.shape[0] - SAMPLE_RATE, SAMPLE_RATE // 2):
                        audio_segment = wav_data[start:start + SAMPLE_RATE]
                        uttid = f'{wav.rstrip(".wav")}_{start:08d}'
                        wavfile.write(
                            os.path.join(processed_dir, f'{uttid}.wav'),
                            SAMPLE_RATE,
                            audio_segment
                        )
                        text_f.write(uttid + ' ' + label + '\n')
                        wav_scp_f.write(uttid + ' ' + os.path.abspath(os.path.join(processed_dir, f'{uttid}.wav')) + '\n')
                        utt2spk_f.write(uttid + ' ' + uttid + '\n')
