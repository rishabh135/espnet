#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=2               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=64                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################

global_dir=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/ # used primarily to handle going in and out of directories especially for espenet2.bin.launch




data_dd=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/partial_data_xvector_speed/original_data
dumpdir=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/partial_data_xvector_speed/dump




# dumpdir=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/${project_name}/dump # Directory to dump features.
# expdir=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/${project_name}/exp # Directory to save experiments.



adversarial_flag="True"
vae_flag="True"
# adv_liststr="asr_adv_asradv"
adv_liststr="recon 600"
# adv_liststr="asr 40 adv 40 asradv 40 reinit_adv 40"


resume_checkpoint=-1
max_epoch=600
batch_bins=24000000
adv_weight=25.0
adv_dropout_out=0.0
adv_dropout_mid=0.0
adv_dropout_inp=0.0
vae_weight_factor=0.8
save_every_epoch=100
vae_annealing_cycle=30
plot_iiter=2
latent_dim=80
accum_grad=2



project_name="vae_v2_april_24_one_speaker_distributed"
experiment_name="latent_dim_80_with_spembs"

expdir=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/${project_name}/${experiment_name}/exp # Directory to dump features.




echo "\n******************************\n"
echo "${project_name}"
echo "${adv_liststr}"
echo "$dumpdir"
echo "$expdir"
echo "********\n Important setting data direcotry  *********** \n"
echo "\n data directory : ${data_dd}  \n"
echo "\n****************************\n"

###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
# project_name=pyt_adversarial_june_7
# project_name=pyt_1


# dumpdir=dump         # Directory to dump features.
# expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts=${data_dd} # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.5 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Ngram model related
use_ngram=false
ngram_exp=
ngram_num=3
use_xvector=
xvector_tool=kaldi
# xvector_model=

# Language model related
use_lm=       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the directory path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false    # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
asr_tag=       # Suffix to the result dir for asr model training.
asr_exp=       # Specify the directory path for ASR experiment.
               # If this option is specified, asr_tag is ignored.
asr_stats_dir= # Specify the directory path for ASR statistics.
asr_config=    # Config for asr model training.
asr_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus.

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
k2_ctc_decoding=true
use_nbest_rescoring=true # use transformer-decoder
                         # and transformer language model for nbest rescoring
num_paths=1000 # The 3rd argument of k2.random_paths.
nll_batch_size=100 # Affect GPU memory usage when computing nll
                   # during nbest rescoring
k2_config=./conf/decode_asr_transformer_with_k2.yaml

use_streaming=false # Whether to use streaming decoding

use_maskctc=false # Whether to use maskctc decoding

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_asr_model=valid.acc.ave.pth # ASR model path for decoding.
                                      # e.g.
                                      # inference_asr_model=train.loss.best.pth
                                      # inference_asr_model=3epoch.pth
                                      # inference_asr_model=valid.acc.best.pth
                                      # inference_asr_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
asr_speech_fold_length=800 # fold_length for speech data during ASR training.
asr_text_fold_length=150   # fold_length for text data during ASR training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    --use_xvector      # Whether to use X-vector (default="${use_xvector}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ASR model related
    --asr_tag          # Suffix to the result dir for asr model training (default="${asr_tag}").
    --asr_exp          # Specify the directory path for ASR experiment.
                       # If this option is specified, asr_tag is ignored (default="${asr_exp}").
    --asr_stats_dir    # Specify the directory path for ASR statistics (default="${asr_stats_dir}").
    --asr_config       # Config for asr model training (default="${asr_config}").
    --asr_args         # Arguments for asr model training (default="${asr_args}").
                       # e.g., --asr_args "--max_epoch 10"
                       # Note that it will overwrite args in asr config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_asr_model # ASR model path for decoding (default="${inference_asr_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    --use_streaming       # Whether to use streaming decoding (default="${use_streaming}").
    --use_maskctc         # Whether to use maskctc decoding (default="${use_streaming}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --asr_speech_fold_length # fold_length for speech data during ASR training (default="${asr_speech_fold_length}").
    --asr_text_fold_length   # fold_length for text data during ASR training (default="${asr_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Use the same text as ASR for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"




# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=${data_dd}/${lang}_token_list
else
    token_listdir=${data_dd}/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi


# Set tag for naming of model directory
if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
    else
        asr_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        asr_tag+="_${lang}_${token_type}"
    else
        asr_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${lm_token_type}"
    else
        lm_tag+="_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${asr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${lang}_${token_type}"
    else
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${asr_exp}" ]; then
    asr_exp="${expdir}/asr_${asr_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
      inference_tag+="_use_k2"
      inference_tag+="_k2_ctc_decoding_${k2_ctc_decoding}"
      inference_tag+="_use_nbest_rescoring_${use_nbest_rescoring}"
    fi
fi



# ========================== Main stages start from here. ==========================

if "${skip_data_prep}"; then
    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi
fi

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for ${data_dd}/${train_set}, ${data_dd}/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: ${data_dd}/${train_set} -> ${data_dd}/${train_set}_sp"
           for factor in ${speed_perturb_factors}; do
               if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                   ${global_dir}/scripts/utils/perturb_data_dir_speed.sh "${factor}" "${data_dd}/${train_set}" "${data_dd}/${train_set}_sp${factor}"
                   _dirs+="${data_dd}/${train_set}_sp${factor} "
               else
                   # If speed factor is 1, same as the original
                   _dirs+="${data_dd}/${train_set} "
               fi
           done
           ${global_dir}/utils/combine_data.sh "${data_dd}/${train_set}_sp" ${_dirs}
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: ${data_dd}/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                ${global_dir}/utils/copy_data_dir.sh --validate_opts --non-print ${data_dd}/"${dset}" "${data_feats}${_suf}/${dset}"
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e ${data_dd}/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments ${data_dd}/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                ${global_dir}/scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "${data_dd}/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank_pitch ]; then
            log "[Require Kaldi] Stage 3: ${feats_type} extract: ${data_dd}/ -> ${data_feats}"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                ${global_dir}/utils/copy_data_dir.sh --validate_opts --non-print ${data_dd}/"${dset}" "${data_feats}${_suf}/${dset}"

                # 2. Feature extract
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
                ${global_dir}/utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

                # 3. Derive the the frame length and feature dimension
                ${global_dir}/scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                # 4. Write feats_dim
                head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                    | cut -d, -f2 > ${data_feats}${_suf}/${dset}/feats_dim

                # 5. Write feats_type
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank ]; then
            log "Stage 3: ${feats_type} extract: ${data_dd}/ -> ${data_feats}"
            log "${feats_type} is not supported yet."
            exit 1

        elif  [ "${feats_type}" = extracted ]; then
            log "Stage 3: ${feats_type} extract: ${data_dd} -> ${data_feats}"
            # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # Generate dummy wav.scp to avoid error by copy_data_dir.sh
                <${data_dd}/"${dset}"/cmvn.scp awk ' { print($1,"<DUMMY>") }' > ${data_dd}/"${dset}"/wav.scp
                ${global_dir}/utils/copy_data_dir.sh --validate_opts --non-print ${data_dd}/"${dset}" "${data_feats}${_suf}/${dset}"

                # Derive the the frame length and feature dimension
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                ${global_dir}/scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - | \
                    awk '{ print $2 }' | cut -d, -f2 > "${data_feats}${_suf}/${dset}/feats_dim"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short ${data_dd}: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do

            # Copy data dir
            ${global_dir}/utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                <"${data_feats}/org/${dset}/utt2num_samples" \
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                        >"${data_feats}/${dset}/utt2num_samples"
                <"${data_feats}/org/${dset}/wav.scp" \
                    ${global_dir}/utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/wav.scp"
            else
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                    | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                <"${data_feats}/org/${dset}/feats.scp" \
                    ${global_dir}/utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            ${global_dir}/utils/fix_data_dir.sh "${data_feats}/${dset}"
        done

        # shellcheck disable=SC2002
        cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        if [ "${token_type}" = bpe ]; then
            log "Stage 5: Generate token_list from ${bpe_train_text} using BPE"

            mkdir -p "${bpedir}"
            # shellcheck disable=SC2002
            cat ${bpe_train_text} | cut -f 2- -d" "  > "${bpedir}"/train.txt

            if [ -n "${bpe_nlsyms}" ]; then
                _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
            else
                _opts_spm=""
            fi

            spm_train \
                --input="${bpedir}"/train.txt \
                --vocab_size="${nbpe}" \
                --model_type="${bpemode}" \
                --model_prefix="${bpeprefix}" \
                --character_coverage=${bpe_char_cover} \
                --input_sentence_size="${bpe_input_sentence_size}" \
                ${_opts_spm}

            {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
            } > "${token_list}"

        elif [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
            log "Stage 5: Generate character level token_list from ${lm_train_text}"

            _opts="--non_linguistic_symbols ${nlsyms_txt}"

            # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
            # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
            ${python} -m espnet2.bin.tokenize_text  \
                --token_type "${token_type}" \
                --input "${data_feats}/lm_train.txt" --output "${token_list}" ${_opts} \
                --field 2- \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --write_vocabulary true \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"

        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
        fi

        # Create word-list for word-LM training
        if ${use_word_lm} && [ "${token_type}" != word ]; then
            log "Generate word level token_list from ${data_feats}/lm_train.txt"
            ${python} -m espnet2.bin.tokenize_text \
                --token_type word \
                --input "${data_feats}/lm_train.txt" --output "${lm_token_list}" \
                --field 2- \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --write_vocabulary true \
                --vocabulary_size "${word_vocab_size}" \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"
        fi

    fi
else
    log "Skip the stages for data preparation"
fi









################################################################################################################################################
# https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/tts1/tts.sh

# Error https://github.com/espnet/espnet/issues/4154  https://github.com/espnet/espnet/issues?q=is%3Aissue+wav-to-duration+is+not+on+your+path
# utils/validate_data_dir.sh: Successfully validated data-directory dump/22k/mfcc/tr_no_dev_phn
# utils/data/get_utt2dur.sh: segments file does not exist so getting durations from wave files
# utils/data/get_utt2dur.sh: could not get utterance lengths from sphere-file headers, using wav-to-duration
# utils/data/get_utt2dur.sh: wav-to-duration is not on your path   
# https://github.com/kaldi-asr/kaldi/issues/4151 seems to be like its a intel MKL library issue


# Extract X-vector
if "${use_xvector}"; then
    echo "stage 3: x-vector extraction"

    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=${data_dd}/mfccdir
    vaddir=${data_dd}/mfcc
    nnet_dir=${dumpdir}/nnet/

    # for name in ${train_set} ${valid_set} ${test_sets} ; do
    #     ${global_dir}/utils/copy_data_dir.sh ${data_dd}/${name} ${data_dd}/${name}_mfcc_16k
    #     ${global_dir}/utils/data/resample_data_dir.sh 16000 ${data_dd}/${name}_mfcc_16k
    #     ${global_dir}/steps/make_mfcc.sh  \
    #         --write-utt2num-frames true \
    #         --mfcc-config ${global_dir}/conf/mfcc.conf \
    #         --nj ${nj} --cmd "$train_cmd" \
    #         ${data_dd}/${name}_mfcc_16k ${expdir}/make_mfcc_16k ${mfccdir}
    #     ${global_dir}/utils/fix_data_dir.sh ${data_dd}/${name}_mfcc_16k
    #     ${global_dir}/steps/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
    #         ${data_dd}/${name}_mfcc_16k ${expdir}/make_vad ${vaddir}
    #     ${global_dir}/utils/fix_data_dir.sh ${data_dd}/${name}_mfcc_16k
    # done

    # # Check pretrained model existence
    # if [ ! -e ${nnet_dir} ]; then
    #     echo "X-vector model does not exist. Download pre-trained model."
    #     wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
    #     tar xvf ./0008_sitw_v2_1a.tar.gz
    #     mv ./0008_sitw_v2_1a/exp/xvector_nnet_1a ${nnet_dir}
    #     rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    # fi
    # # Extract x-vector
    # for name in ${train_set} ${valid_set} ${test_sets}; do
    #     ${global_dir}/../../../tools/kaldi/egs/sre08/v1/sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
    #         ${nnet_dir} ${data_dd}/${name}_mfcc_16k \
    #         ${nnet_dir}/xvectors_${name}
    # done

    # Update json
    # for name in ${train_set} ${valid_set} ${test_sets}; do
    #     ./local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    # done





    ################################################################################################################################################
    # https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/tts1/tts.sh
    # Filter x-vector 
    # /srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/data_with_speed_version_xvector/dump/nnet/xvectors_train_clean_100_sp/xvector.scp

    # log " Stage 4+ xvector filtering"
    # if "${use_xvector}"; then
    #     cp "${dumpdir}/xvector/${dset}"/xvector.{scp,scp.bak}
    #     <"${dumpdir}/xvector/${dset}/xvector.scp.bak" \
    #         ${global_dir}/utils/filter_scp.pl "${data_feats}/${dset}/wav.scp"  \
    #         >"${dumpdir}/xvector/${dset}/xvector.scp"
    # fi
    ################################################################################################################################################


fi
################################################################################################################################################








# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if "${use_lm}"; then
        if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
            log "Stage 6: LM collect stats: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            # 1. Split the key file
            _logdir="${lm_stats_dir}/logdir"
            mkdir -p "${_logdir}"
            # Get the minimum number among ${nj} and the number lines of input files
            _nj=$(min "${nj}" "$(<${data_feats}/lm_train.txt wc -l)" "$(<${lm_dev_text} wc -l)")

            key_file="${data_feats}/lm_train.txt"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/train.${n}.scp"
            done
            # shellcheck disable=SC2086
            ${global_dir}/utils/split_scp.pl "${key_file}" ${split_scps}

            key_file="${lm_dev_text}"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/dev.${n}.scp"
            done
            # shellcheck disable=SC2086
            ${global_dir}/utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Generate run.sh
            log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
            mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"


            log "\n *********************** changing directories Stage 6 Language modeling **************************\n"
            cd "${global_dir}"
            cd "../../../"

            # 3. Submit jobs
            log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m espnet2.bin.lm_train \
                    --collect_stats true \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --train_data_path_and_name_and_type "${data_feats}/lm_train.txt,text,text" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --train_shape_file "${_logdir}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/dev.JOB.scp" \
                    --output_dir "${_logdir}/stats.JOB" \
                    ${_opts} ${lm_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

            # 4. Aggregate shape files
            _opts=
            for i in $(seq "${_nj}"); do
                _opts+="--input_dir ${_logdir}/stats.${i} "
            done
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

            # Append the num-tokens at the last dimensions. This is used for batch-bins count
            <"${lm_stats_dir}/train/text_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/train/text_shape.${lm_token_type}"

            <"${lm_stats_dir}/valid/text_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/valid/text_shape.${lm_token_type}"
        fi


        if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            log "Stage 7: LM Training: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            if [ "${num_splits_lm}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                      --scps "${data_feats}/lm_train.txt" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
                      --num_splits "${num_splits_lm}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.txt,text,text "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${data_feats}/lm_train.txt,text,text "
                _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
            fi

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

            log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 7 using this script"
            mkdir -p "${lm_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

            log "LM training started... log: '${lm_exp}/train.log'"
            if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
                # SGE can't include "/" in a job name
                jobname="$(basename ${lm_exp})"
            else
                jobname="${lm_exp}/train.log"
            fi





            log "\n *********************** changing directories Stage 7 Language modeling **************************\n"
            cd "${global_dir}"
            cd "../../../"


            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                --cmd "${cuda_cmd} --name ${jobname}" \
                --log "${lm_exp}"/train.log \
                --ngpu "${ngpu}" \
                --num_nodes "${num_nodes}" \
                --init_file_prefix "${lm_exp}"/.dist_init_ \
                --multiprocessing_distributed true -- \
                ${python} -m espnet2.bin.lm_train \
                    --ngpu "${ngpu}" \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --adv_flag "" \
                    --vae_flag "${vae_flag}" \
                    --project_name "${project_name}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
                    --fold_length "${lm_fold_length}" \
                    --resume true \
                    --output_dir "${lm_exp}" \
                    ${_opts} ${lm_args}

        fi


        if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
            log "Stage 8: Calc perplexity: ${lm_test_text}"
            _opts=
            # TODO(kamo): Parallelize?
            log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
            # shellcheck disable=SC2086
            ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
                ${python} -m espnet2.bin.lm_calc_perplexity \
                    --ngpu "${ngpu}" \
                    --data_path_and_name_and_type "${lm_test_text},text,text" \
                    --train_config "${lm_exp}"/config.yaml \
                    --model_file "${lm_exp}/${inference_lm}" \
                    --output_dir "${lm_exp}/perplexity_test" \
                    ${_opts}
            log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

        fi

        ################################################################################################################################################
        # https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/tts1/tts.sh
        log "Stage 8-B : TTS collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi



        # https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/tts1/tts.sh
        # /srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/data_with_speed_version_xvector/dump/nnet/xvectors_train_clean_100_sp/spk_xvector.scp
        # if "${use_xvector}"; then
        #     _xvector_train_dir="${dumpdir}/xvector/${train_set}"
        #     log "${_xvector_train_dir}"
        #     _xvector_valid_dir="${dumpdir}/xvector/${valid_set}"
        #     _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/spk_xvector.scp,spembs,kaldi_ark "
        #     _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/spk_xvector.scp,spembs,kaldi_ark "
        # fi


        # if "${use_sid}"; then
        #     _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2sid,sids,text_int "
        #     _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2sid,sids,text_int "
        # fi
        ####################################################################################################################################################################################

    else
        log "Stage 6-8: Skip lm-related stages: use_lm=${use_lm}"
    fi


    if "${use_ngram}"; then
        mkdir -p ${ngram_exp}
    fi
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        if "${use_ngram}"; then
            log "Stage 9: Ngram Training: train_set=${data_feats}/lm_train.txt"
            cut -f 2- -d " " ${data_feats}/lm_train.txt | ${global_dir}/kenlm/build/bin/lmplz -S "20%" --discount_fallback -o ${ngram_num} - >${ngram_exp}/${ngram_num}gram.arpa
            ${global_dir}/kenlm/build/bin/build_binary -s ${ngram_exp}/${ngram_num}gram.arpa ${ngram_exp}/${ngram_num}gram.bin
        else
            log "Stage 9: Skip ngram stages: use_ngram=${use_ngram}"
        fi
    fi


    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        _asr_train_dir="${data_feats}/${train_set}"
        _asr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: ASR collect stats: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi

        _feats_type="$(<${_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${asr_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_asr_train_dir}/${_scp} wc -l)" "$(<${_asr_valid_dir}/${_scp} wc -l)")

        key_file="${_asr_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        ${global_dir}/utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_asr_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        ${global_dir}/utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${asr_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${asr_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${asr_stats_dir}/run.sh"; chmod +x "${asr_stats_dir}/run.sh"

        # 3. Submit jobs
        log "ASR collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.


        log "\n *********************** changing directories Stage 10    **************************\n"
        cd "${global_dir}"
        cd "../../../"

        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.asr_train \
                --collect_stats true \
                --use_preprocessor true \
                --project_name "${project_name}" \
                --adv_flag "${adversarial_flag}" \
                --vae_flag "${vae_flag}" \
                --max_epoch "${max_epoch}" \
                --batch_bins "${batch_bins}" \
                --adv_liststr "${adv_liststr}" \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --allow_variable_data_keys true \
                --g2p "${g2p}" \
                --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_asr_train_dir}/text,text,text" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                --train_data_path_and_name_and_type "${_asr_train_dir}/utt2spkid.txt,spkid,text_int" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${asr_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }


        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done


        # echo "*********  use xvector addding spembs stage 10 "
        # echo " "
        # if "${use_xvector}"; then
        #     _xvector_train_dir="${dumpdir}/xvector/${train_set}"
        #     _xvector_valid_dir="${dumpdir}/xvector/${valid_set}"
        #     _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
        #     _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
        # fi





        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${asr_stats_dir}/train/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${asr_stats_dir}/train/text_shape.${token_type}"

        <"${asr_stats_dir}/valid/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${asr_stats_dir}/valid/text_shape.${token_type}"
    fi


    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        _asr_train_dir="${data_feats}/${train_set}"
        _asr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 11: ASR Training: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi

        _feats_type="$(<${_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((asr_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${asr_speech_fold_length}"
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_asr_train_dir}/${_scp}" \
                      "${_asr_train_dir}/text" \
                      "${asr_stats_dir}/train/speech_shape" \
                      "${asr_stats_dir}/train/text_shape.${token_type}" \
                  --num_splits "${num_splits_asr}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.${token_type} "
        fi





        echo "*********  use xvector adding stage 11 "
        echo " "
        if "${use_xvector}"; then
            _xvector_train_dir="${dumpdir}/raw/${train_set}"
            _xvector_valid_dir="${dumpdir}/raw/${valid_set}"
            echo "${_xvector_train_dir}"
            echo "${_xvector_valid_dir}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
        fi




        log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 11 using this script"
        mkdir -p "${asr_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${asr_exp}/run.sh"; chmod +x "${asr_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "ASR training started... log: '${asr_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${asr_exp})"
        else
            jobname="${asr_exp}/train.log"
        fi

        log "\n *********************** changing directories Stage 11 **************************\n"
        cd "${global_dir}"
        cd "../../../"

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${asr_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${asr_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.asr_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --adv_liststr "${adv_liststr}" \
                --adv_flag "${adversarial_flag}" \
                --vae_flag "${vae_flag}" \
                --adv_dropout_out "${adv_dropout_out}" \
                --adv_dropout_inp "${adv_dropout_inp}" \
                --adv_dropout_mid "${adv_dropout_mid}" \
                --max_epoch "${max_epoch}" \
                --batch_bins "${batch_bins}" \
                --project_name "${project_name}" \
                --resume_from_checkpoint "${resume_checkpoint}" \
                --adv_loss_weight "${adv_weight}" \
                --plot_iiter "${plot_iiter}" \
                --accum_grad "${accum_grad}" \
                --latent_dim "${latent_dim}" \
                --save_every_epoch "${save_every_epoch}" \
                --vae_annealing_cycle "${vae_annealing_cycle}" \
                --vae_weight_factor "${vae_weight_factor}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --allow_variable_data_keys true \
                --train_data_path_and_name_and_type "${_asr_train_dir}/utt2spkid.txt,spkid,text_int" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/utt2spkid.txt,spkid,text_int" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
                --valid_shape_file "${asr_stats_dir}/valid/text_shape.${token_type}" \
                --resume true \
                --init_param ${pretrained_model} \
                --ignore_init_mismatch ${ignore_init_mismatch} \
                --fold_length "${_fold_length}" \
                --fold_length "${asr_text_fold_length}" \
                --output_dir "${asr_exp}" \
                ${_opts} ${asr_args}

    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    asr_exp="${expdir}/${download_model}"
    mkdir -p "${asr_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${asr_exp}/config.txt"

    # Get the path of each file
    _asr_model_file=$(<"${asr_exp}/config.txt" sed -e "s/.*'asr_model_file': '\([^']*\)'.*$/\1/")
    _asr_train_config=$(<"${asr_exp}/config.txt" sed -e "s/.*'asr_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_asr_model_file}" "${asr_exp}"
    ln -sf "${_asr_train_config}" "${asr_exp}"
    inference_asr_model=$(basename "${_asr_model_file}")

    if [ "$(<${asr_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${asr_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${asr_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Decoding: training_dir=${asr_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi
        if "${use_ngram}"; then
             _opts+="--ngram_file ${ngram_exp}/${inference_ngram}"
        fi

        # 2. Generate run.sh
        log "Generate '${asr_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${asr_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${asr_exp}/${inference_tag}/run.sh"; chmod +x "${asr_exp}/${inference_tag}/run.sh"
        if "${use_k2}"; then
          # Now only _nj=1 is verified if using k2
          asr_inference_tool="espnet2.bin.asr_inference_k2"

          _opts+="--is_ctc_decoding ${k2_ctc_decoding} "
          _opts+="--use_nbest_rescoring ${use_nbest_rescoring} "
          _opts+="--num_paths ${num_paths} "
          _opts+="--nll_batch_size ${nll_batch_size} "
          _opts+="--k2_config ${k2_config} "
        else
          if "${use_streaming}"; then
              asr_inference_tool="espnet2.bin.asr_inference_streaming"
          elif "${use_maskctc}"; then
              asr_inference_tool="espnet2.bin.asr_inference_maskctc"
          else
              asr_inference_tool="espnet2.bin.asr_inference"
          fi
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${asr_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            if "${use_k2}"; then
              # Now only _nj=1 is verified if using k2
              _nj=1
            else
              _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            fi

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            ${global_dir}/utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
            log "\n *********************** changing directories Stage 12 **************************\n"
            cd "${global_dir}"
            cd "../../../"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                ${python} -m ${asr_inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --asr_train_config "${asr_exp}"/config.yaml \
                    --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/asr_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                  for i in $(seq "${_nj}"); do
                      cat "${_logdir}/output.${i}/1best_recog/${f}"
                  done | sort -k1 >"${_dir}/${f}"
                fi
            done
        done
    fi


    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "Stage 13: Scoring"
        log "\n *********************** changing directories Stage 13 **************************\n"
        cd "${global_dir}"
        cd "../../../"
        if [ "${token_type}" = phn ]; then
            log "Error: Not implemented for token_type=phn"
            exit 1
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${asr_exp}/${inference_tag}/${dset}"

            for _type in cer wer ter; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    # Tokenize text to word level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"


                elif [ "${_type}" = cer ]; then
                    # Tokenize text to char level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                elif [ "${_type}" = ter ]; then
                    # Tokenize text using BPE
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  --cleaner "${cleaner}" \
                                ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                fi

                ${global_dir}../../../tools/kaldi/tools/sctk-20159b5/bin/sclite  \
                    ${score_opts} \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done
        done

        [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${asr_exp}"

        # Show results in Markdown syntax
        ${global_dir}/scripts/utils/show_asr_result.sh "${asr_exp}" > "${asr_exp}"/RESULTS.md
        cat "${asr_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${asr_exp}/${asr_exp##*/}_${inference_asr_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 14: Pack model: ${packed_model}"

        _opts=
        if "${use_lm}"; then
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
            _opts+="--option ${lm_exp}/perplexity_test/ppl "
            _opts+="--option ${lm_exp}/images "
        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            _opts+="--option ${asr_stats_dir}/train/feats_stats.npz "
        fi
        if [ "${token_type}" = bpe ]; then
            _opts+="--option ${bpemodel} "
        fi
        if [ "${nlsyms_txt}" != none ]; then
            _opts+="--option ${nlsyms_txt} "
        fi

        log "\n *********************** changing directories Stage 14 **************************\n"
        cd "${global_dir}"
        cd "../../../"
        
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack asr \
            --asr_train_config "${asr_exp}"/config.yaml \
            --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
            ${_opts} \
            --option "${asr_exp}"/RESULTS.md \
            --option "${asr_exp}"/RESULTS.md \
            --option "${asr_exp}"/images \
            --outpath "${packed_model}"
    fi
fi

if ! "${skip_upload}"; then
    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        log "Stage 15: Upload model to Zenodo: ${packed_model}"
        log "Warning: Upload model to Zenodo will be deprecated. We encourage to use Hugging Face"

        # To upload your model, you need to do:
        #   1. Sign up to Zenodo: https://zenodo.org/
        #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
        #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="git checkout $(git show -s --format=%H)"

        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${asr_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${asr_exp}"/RESULTS.md)</code></pre></li>
<li><strong>ASR config</strong><pre><code>$(cat "${asr_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${asr_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stage"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
	    exit 1
        log "Stage 16: Upload model to HuggingFace: ${hf_repo}"

        gitlfs=$(git lfs --version 2> /dev/null || true)
        [ -z "${gitlfs}" ] && \
            log "ERROR: You need to install git-lfs first" && \
            exit 1

        dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
        [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="git checkout $(git show -s --format=%H)"
        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=automatic-speech-recognition
        # shellcheck disable=SC2034
        espnet_task=ASR
        # shellcheck disable=SC2034
        task_exp=${asr_exp}
        eval "echo \"$(cat ${global_dir}/scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

        this_folder=${PWD}
        cd ${dir_repo}
        if [ -n "$(git status --porcelain)" ]; then
            git add .
            git commit -m "Update model"
        fi
        git push
        cd ${this_folder}
    fi
else
    log "Skip the uploading to HuggingFace stage"
fi

