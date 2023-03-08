#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# train_set="train_clean_100"
# valid_set="dev_clean"
# test_sets="test_clean test_other dev_clean dev_other"


train_set="dev_clean"
valid_set="dev_clean"
test_sets="test_clean test_other dev_clean dev_other"

asr_tag=conformer_lr2e-3_warmup15k_amp_nondeterministic


###################################################################################################################################################################################################




data_dd=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/data_xvector_overfit/original_data


project_name="vae_overfit_march_9_recon_decoder_updated"


# project_name="vae_feb_21_recon_100"

# project_name="vae_lsoftmax_feb_28_beta_factor_0.8_adv_weight_25"


###################################################################################################################################################################################################
###################################################################################################################################################################################################
# asr_config=conf/train_asr.yaml
# inference_config=conf/decode_asr.yaml

# srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/${experiment_n}
# experiment_n=asr_lmt_trigram_wo_adv
# experiment_n=pyt_adversarial_june_7 # name of the experiment, just change it to create differnet folders
# experiment_n=pyt_adversarial_june_7 # name of the experiment, just change it to create differnet folders


# data_dd=/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/fresh_libri_100/${experiment_n}/data # determines all the files creating folder as in the data folder

# data_dd=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/data


asr_config=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/conf/train_asr_vae.yaml
inference_config=/home/rgupta/dev/espnet/egs2/librispeech_100/asr1/conf/decode_asr.yaml


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
./vae_asr_overfit_skip_data_lm.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
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
    --bpe_train_text "${data_dd}/${train_set}/text" "$@"