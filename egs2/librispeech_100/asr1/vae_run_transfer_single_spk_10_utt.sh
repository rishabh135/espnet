#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# train_set="train_clean_100"
# valid_set="dev_clean"
# test_sets="test_clean test_other dev_clean dev_other"


train_set="train_clean_1_spk_1_utt"
valid_set="train_clean_1_spk_1_utt"
test_sets="train_clean_1_spk_1_utt"

asr_tag=conformer_lr2e-3_warmup15k_amp_nondeterministic


###################################################################################################################################################################################################




data_dd=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/partial_data_xvector_speed/original_data





project_name="vae_xtransfer_june_9_utt_10_asrlr_010"




###################################################################################################################################################################################################
###################################################################################################################################################################################################



asr_config=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/conf/train_asr_vae_single_spk.yaml
inference_config=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/conf/decode_asr.yaml


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
./vae_asr_transfer_single_spk_10_utt.sh \
    --skip_data_prep true \
    --skip_train false \
    --skip_eval true \
    --lang en \
    --ngpu 1 \
    --nj 64 \
    --use_xvector true \
    --inference_nj 32 \
    --nbpe 5000 \
    --max_wav_duration 30 \
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
