import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.contrib import tqdm


def main():
    """Load the model, generate kernel and bandpass plots."""
    global_dir= "/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/"
    adversarial_flag="True"
    project_name="nancy_july_27_data_prep_adv"
    experiment_name="odim_585_asradvasradv10+asr10"
    with_sp = ""
    utt2spkfile = "{}/{}/data/train_clean_100{}/utt2spk".format(global_dir,project_name, with_sp) 
    spk2genderfile = "{}/{}/data/train_clean_100{}/spk2gender".format(global_dir, project_name, with_sp) 
    output_file =  "{}/{}/data/train_clean_100_sp{}/utt2spkid.txt".format(global_dir, project_name, with_sp)  
    
    main_map = {}
    utt_index = {}
    
    with open( spk2genderfile, "r+") as work_data:
        # File object is now open.
        # Do stuff with the file:
        inde = 0
        for line in work_data:
            spk, gend = line.split(" ")
            # print(spk, gend)
            main_map[str(spk)] = inde
            inde += 1
    

    keylist = list(main_map.keys()) 
    print(keylist[0])
    print(type(keylist[0]))
    print("\n\n*********************************")
    with open( utt2spkfile, "r+") as work_data:
        # File object is now open.
        # Do stuff with the file:
        inde = 0
        for line in work_data:
            utt, spkid = line.strip("\n").split(" ")
            utt_index[str(utt)] = main_map[str(spkid)]
            print(utt, spkid)
            
    print("\n*********************************\n")

    with open( output_file, "w") as work_data:
        # File object is now open.
        # Do stuff with the file:
        inde = 0
        for key, value in utt_index.items():
            print("{} {}".format(key, value), file=work_data)

            # utt, spkid = line.split(" ")
            # utt_index[str(utt)] = main_map[str(spkid)]
            # print(utt, spkid)

    # parser = get_parser()
    # args = parser.parse_args(argv)





if __name__=="__main__":
    main()