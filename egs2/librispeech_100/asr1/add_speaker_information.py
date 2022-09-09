import argparse
import logging
import os
import sys
# from pathlib import Path

import pathlib
import numpy as np
import torch
from tqdm.contrib import tqdm


# experiment_name="odim_585_asradvasradv10+asr10"
# dump_folder = "nancy_july_28_data_prep_adv/dump/raw/train_clean_100/"
    




def main():


    """ setting paths plots."""
    in_global_dir= "/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/nancy_v2_sep_9/data"
    out_global_dir= "/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/nancy_v2_sep_9/odim_251_single_lr/dump/raw"
    # project_name="nancy_july_29_data_prep_adv"
    

    with_sp_list = [ "train_clean_100", "train_clean_100_sp", "dev"]
    for with_sp in with_sp_list:
        utt2spkfile = "{}/{}/utt2spk".format(in_global_dir, with_sp) 
        spk2genderfile = "{}/{}/spk2gender".format(in_global_dir, with_sp) 
        output_file =  "{}/{}/utt2spkid.txt".format(out_global_dir,with_sp)  
        
        pathlib.Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        main_map = {}
        utt_index = {}
        with open( spk2genderfile, "r+") as work_data:
            # File object is now open.
            # Do stuff with the file:
            inde = 0
            for line in work_data:
                spk, gend = line.split(" ")
                if(spk.isnumeric()):
                    print("Inside spk {}  gend {} ".format(spk, gend))
                    main_map[str(spk)] = inde
                    inde += 1
            

        keylist = list(main_map.keys()) 
        # print(keylist[0])
        # print(type(keylist[0]))
        print("\n\n*********************************")
        with open( utt2spkfile, "r+") as work_data:
            inde = 0
            for line in work_data:
                utt, spkid = line.strip("\n").split(" ")
                if(not spkid.isnumeric()):
                    spkid = spkid.split("-")[-1]
                utt_index[str(utt)] = main_map[str(spkid)]
                print("Inside utt {}  spkid {} ".format(utt, spkid))
                    
                
        print("\n*********************************\n")
        print(" {} \n {} \n {}\n ".format(utt2spkfile,spk2genderfile, output_file))
        with open( output_file, "w") as work_data:
            # File object is now open.
            # Do stuff with the file:
            inde = 0
            for key, value in utt_index.items():
                print("{} {}".format(key, value), file=work_data)

                # utt, spkid = line.split(" ")
                # utt_index[str(utt)] = main_map[str(spkid)]
                # print(utt, spkid)






if __name__=="__main__":
    main()