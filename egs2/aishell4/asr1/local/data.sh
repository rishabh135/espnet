#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



#################################################################
#####             Downloading their git          ################
#################################################################


# Github AISHELL4 : https://github.com/felixfuyihui/AISHELL-4.git
FOLDER=AISHELL-4   # voir à quel endroit le mettre ! 
URL=https://github.com/felixfuyihui/AISHELL-4.git

if [ ! -d "$FOLDER" ] ; then
    git clone "$URL" "$FOLDER"
    log "git successfully downloaded"
fi

#pip install -r "$FOLDER"/requirements.txt   # ATTENTION JE CHANGE LEURS REQUIERMENTS POUR METTRE SENTENCE PIECE 0.1.94 AU LIEU DE 0.1.91 QUI FAIT TOUT BUGUER
# IDEM POUR TORCH JE MET 1.9, voir le vrai git pour les versions d'origine ! 
# rappel : attention, j'ai modifié leur Git  --> oui, il faut que je le fork


#################################################################
#####            Downloading data and producing lists      ##############
#################################################################



if false ; then

    for room_name in "train_L" "train_M" "train_S" 
    do 

        wget https://www.openslr.org/resources/111/$room_name.tar.gz -P ${AISHELL4}/  
        
        
        tar -xzvf ${AISHELL4}/"$room_name".tar.gz -C ${AISHELL4}/
        

        # after that untar step, you have one folder "$room_name" with two subfolders : 
        #   - wav : a list of .flac audio files, each audio file is a conference meeting of about 30 minutes 
        #   - TextGrid : a list of .TextGrid and .rttm files 

        # then you have to produce a list of the names of the files located in the "$room_name"/wav/ directory 
        # list should be like : 
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/wav/20200707_L_R001S01C01.flac
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/wav/20200709_L_R002S06C01.flac
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/wav/20200707_L_R001S04C01.flac
        # ...

        rm  ${AISHELL4}/$room_name/wav_list.txt
        FILES="$PWD/${AISHELL4}/$room_name/wav/*"
        for f in $FILES
        do
            echo "$f" >> ${AISHELL4}/$room_name/wav_list.txt
        done



        # then you have to produce a list of the names of the .TextGrid files located in the "$room_name"/textgrid/ directory 
        # list should be like : 
        #/ocean/projects/cis210027p/shared/corpora/AISHELL4/train_L/TextGrid/textgrid_list/20200706_L_R001S08C01.TextGrid
        # ...

        rm ${AISHELL4}/$room_name/TextGrid_list.txt
        FILES="$PWD/${AISHELL4}/$room_name/TextGrid/*.TextGrid"
        for f in $FILES
        do
            echo "$f" >> ${AISHELL4}/$room_name/TextGrid_list.txt
        done

    done
fi


#################################################################
#####            Join train_L, train_M and train_S       ########
#################################################################

if false; then 
    mkdir ${AISHELL4}/full_train
    rm ${AISHELL4}/full_train/wav_list.txt
    rm ${AISHELL4}/full_train/TextGrid_list.txt

    for r in "train_L" "train_M" "train_S" 
    do 
        cat ${AISHELL4}/$r/TextGrid_list.txt >> ${AISHELL4}/full_train/TextGrid_list.txt
        cat ${AISHELL4}/$r/wav_list.txt >> ${AISHELL4}/full_train/wav_list.txt
    done
fi



#################################################################
#####            ground truth for asr, using aishell4 github     ##############
#################################################################


wav_list_aishell4=${AISHELL4}/full_train/wav_list.txt
text_grid_aishell4=${AISHELL4}/full_train/TextGrid_list.txt

output_folder=$PWD/data/

if false ; then 

    log "generating asr training data ..."
    log "(this can take some time)"
    rm -rf "$output_folder"
 
    python "$FOLDER"/data_preparation/generate_asr_trainingdata.py  --output_dir "$output_folder" --mode train --aishell4_wav_list "$wav_list_aishell4" --textgrid_list "$text_grid_aishell4" || log "ca a pas marché" ;

    log "asr training data generated."

fi
 



#################################################################
#####     creating wav.scp from output/train/wav directory    ##############
#################################################################


if false ; then 
    rm $output_folder/train/wav.scp
    FILES="$output_folder/train/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1) 
        echo "$g" "$f" >> $output_folder/train/wav.scp
    done

fi


#################################################################
#####            creating utt2spk and spk2utt  ########
#################################################################

if false ; then 
    rm $output_folder/train/utt2spk
    FILES="$output_folder/train/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1) 
        echo "$g" "$g"  >> $output_folder/train/utt2spk  # we put speaker_id = utt_id
    done


fi 







#################################################################
#####            sort and fix the data  ########
#################################################################


if false ; then
    log "sorting files ... "
    sort data/train/utt2spk -o data/train/utt2spk
    # creating spk2utt from utt2spk
    rm $output_folder/train/spk2utt
    utils/utt2spk_to_spk2utt.pl $output_folder/train/utt2spk > $output_folder/train/spk2utt
    sort data/train/wav.scp -o data/train/wav.scp
    sort data/train/text -o data/train/text
    log "files sorted"

    # then, removing empty lines

    log "fixing files ..."
    ./utils/fix_data_dir.sh data/train
    log "files fixed"
fi


########################## generate the nlsyms.txt list 

cat data/train/text | perl -pe 's/(\<[^\>\<]+\>)/$1\n/g' | perl -pe 's/(\<[^\>\<]+\>)/\n$1/' | grep ^\<.*\>$ | sort -u > data/nlsyms.txt



if false ; then 
    log "random shuffling to prepare dev and test sets ..."

    get_seeded_random()
        {
        seed="$1"
        openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
            </dev/zero 2>/dev/null
        }

    shuf  --random-source=<(get_seeded_random 76) data/train/utt2spk  -o data/train/utt2spk
    shuf  --random-source=<(get_seeded_random 76) data/train/wav.scp  -o data/train/wav.scp
    shuf  --random-source=<(get_seeded_random 76) data/train/text  -o data/train/text

fi 

if false ; then 
    log "selecting lines for train, dev and test ..."

    utils/subset_data_dir.sh --first data/train 600 data/dev
    n=$(($(wc -l < data/train/text) - 600))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev

fi


if false ; then 
    log "resorting the files ..."
    log "train ..."
    sort data/train_nodev/utt2spk -o data/train_nodev/utt2spk
    utils/utt2spk_to_spk2utt.pl data/train_nodev/utt2spk > data/train_nodev/spk2utt
    sort data/train_nodev/wav.scp -o data/train_nodev/wav.scp
    sort data/train_nodev/text -o data/train_nodev/text
    log "files sorted"
    log "test ..."
    sort data/dev/utt2spk -o data/dev/utt2spk
    utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
    sort data/dev/wav.scp -o data/dev/wav.scp
    sort data/dev/text -o data/dev/text
    log "files sorted"


fi 






#################################################################
#####      Combining with aishell1 data  (train only for now)   
#################################################################


# pay attention : sorting issues with utt2spk :  (fix this by making speaker-ids prefixes of utt-ids)

if true ; then 
    
    aishell1_data=/ocean/projects/cis210027p/berrebbi/espnet/egs2/aishell/asr1/data/train
    aishell4_data=data/train_nodev

    u2s=/ocean/projects/cis210027p/berrebbi/espnet/egs2/aishell/asr1/data/train/utt2spk
    awk 'BEGIN {FS=" "; OFS="\n"}; {print $1" "$1}' $u2s > /ocean/projects/cis210027p/berrebbi/espnet/egs2/aishell/asr1/data/train/utt2spk2

    mv /ocean/projects/cis210027p/berrebbi/espnet/egs2/aishell/asr1/data/train/utt2spk2 /ocean/projects/cis210027p/berrebbi/espnet/egs2/aishell/asr1/data/train/utt2spk
    
    utils/combine_data.sh data/combined_aishell_dir/train $aishell1_data $aishell4_data 

    
    sort data/combined_aishell_dir/train/utt2spk -o data/combined_aishell_dir/train/utt2spk
    utils/utt2spk_to_spk2utt.pl data/combined_aishell_dir/train/utt2spk > data/combined_aishell_dir/train/spk2utt
    sort data/combined_aishell_dir/train/wav.scp -o data/combined_aishell_dir/train/wav.scp
    sort data/combined_aishell_dir/train/text -o data/combined_aishell_dir/train/text

    wc -l data/combined_aishell_dir/train/*


fi 


#################################################################
#####            test                                    ########
#################################################################



if false ; then

    #wget https://www.openslr.org/resources/111/test.tar.gz -P ${AISHELL4}/  
    
    
    #tar -xzvf ${AISHELL4}/test.tar.gz -C ${AISHELL4}/
    

    rm  ${AISHELL4}/test/wav_list.txt
    FILES="$PWD/${AISHELL4}/test/wav/*"
    for f in $FILES
    do
        echo "$f" >> ${AISHELL4}/test/wav_list.txt
    done


    rm ${AISHELL4}/test/TextGrid_list.txt
    FILES="$PWD/${AISHELL4}/test/TextGrid/*.TextGrid"
    for f in $FILES
    do
        echo "$f" >> ${AISHELL4}/test/TextGrid_list.txt
    done

fi




#################################################################
#####            ground truth for asr, using aishell4 github     ##############
#################################################################


wav_list_aishell4_test=${AISHELL4}/test/wav_list.txt
text_grid_aishell4_test=${AISHELL4}/test/TextGrid_list.txt

output_folder=$PWD/data/test/

if false ; then 

    log "generating asr test data ..."
    log "(this can take some time)"
    rm -rf "$output_folder"
 
    #python "$FOLDER"/data_preparation/generate_asr_trainingdata.py  --output_dir "$output_folder" --mode train --aishell4_wav_list "$wav_list_aishell4_test" --textgrid_list "$text_grid_aishell4_test" || log "ca a pas marché" ;

    log "asr test data generated."

    python "$FOLDER"/data_preparation/generate_nospk_testdata.py --wav_list $wav_list_aishell4_test --textgrid_list $text_grid_aishell4_test --output_dir $output_folder 

    #mv $output_folder/train/* $output_folder/
    #rm -r $output_folder/train

fi
 



#################################################################
#####     creating wav.scp from output/train/wav directory    ##############
#################################################################


if false ; then 
    rm $output_folder/wav.scp
    FILES="$output_folder/wav/*"
    for f in $FILES
    do

        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1) 
        echo "$g" "$f" >> $output_folder/wav.scp
    done

fi



#################################################################
#####            creating utt2spk and spk2utt  ########
#################################################################

if false ; then 
    rm $output_folder/utt2spk
    FILES="$output_folder/wav/*"
    for f in $FILES
    do
        g=$(echo $f | cut -d'/' -f 14 | cut -d'.' -f 1) 
        echo "$g" "$g"  >> $output_folder/utt2spk  # we put speaker_id = utt_id
    done

    # creationg spk2utt from utt2spk
    rm $output_folder/spk2utt
    utils/utt2spk_to_spk2utt.pl $output_folder/utt2spk > $output_folder/spk2utt
fi 




#################################################################
#####            sort and fix the data  ########
#################################################################


if false ; then
    log "sorting files ... "
    sort data/test/utt2spk -o data/test/utt2spk
    sort data/test/wav.scp -o data/test/wav.scp
    sort data/test/text -o data/test/text
    log "files sorted"

    # then, removing empty lines , IL FAUDRA TROUVER PQ JAI DES EMPTY LINES CEST CHELOU

    log "fixing files ..."
    ./utils/fix_data_dir.sh data/test
    log "files fixed"
fi

