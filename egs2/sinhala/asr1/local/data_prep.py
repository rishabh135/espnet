#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [hyper_root]")
    sys.exit(1)
hyper_root = sys.argv[1]

dir_dict = {
    "train": "train.csv",
    "valid": "validation.csv",
    "test": "test.csv",
}

for x in dir_dict:
    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", x, "transcript"), "w"
    ) as transcript_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        transcript_df = pd.read_csv(os.path.join(hyper_root, "data", dir_dict[x]))
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            print(row[4], " ".join([ch for ch in row[3]]))

            words = row[4].replace(" ", "_") + " " + " ".join([ch for ch in row[3]])
            print(words)
            path_arr = row[1].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            # print(utt_id + " " + words + "\n")
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + hyper_root + "/" + row[1] + "\n")
            utt2spk_f.write(utt_id + " " + row[2] + "\n")
