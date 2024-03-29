#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_tag=conformer_lr2e-3_warmup15k_amp_nondeterministic


###################################################################################################################################################################################################
data_dd=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/data_with_speed_version_2/original_data



project_name="v9_better_lstm_jan_12_160_dropout_only_mid_0.02"


###################################################################################################################################################################################################
###################################################################################################################################################################################################


asr_config=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/conf/train_asr_modified.yaml
inference_config=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/conf/decode_asr.yaml


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################

./v9_dropout_asr_weight_modified_skip_data_lm.sh \
    --skip_data_prep true \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 1 \
    --nj 32 \
    --inference_nj 32 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "${data_dd}/${train_set}/text" \
    --bpe_train_text "${data_dd}/${train_set}/text" "$@" 
    

