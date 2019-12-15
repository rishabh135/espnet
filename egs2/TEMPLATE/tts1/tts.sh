#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --dev-set <dev_set_name> --eval_sets <eval_set_names> --srctexts <srctexts>

Options:
    --stage (int): Processes starts from the specifed stage.
    --stop-stage (int): Processes is stopped at the specifed stage.
    --ngpu (int): The number of gpus
    --nj (int): The number of parallel jobs
    --decode-nj (int): The number of parallel jobs for decoding
    --gpu-decode (bool): Use gpu for decoding.
EOF
)
SECONDS=0

# general configuration
stage=1          # stage to start.
stop_stage=100   # stage to stop.
ngpu=0           # number of gpus ("0" uses cpu, otherwise use gpu).
nj=50            # numebr of parallel jobs.
decode_nj=50     # number of parallel jobs in decoding.
gpu_decode=false # whether to perform gpu decoding.
dumpdir=dump     # directory to dump features.
expdir=exp       # directory to save experiments.

# feature extraction related
feats_type=fbank  # fbank or stft or raw.
audio_format=flac # audio format (only in feats_type=raw).
fs=16000          # sampling rate.
fmax=80           # maximum frequency.
fmin=7600         # minimum frequency.
n_mels=80         # number of mel basis.
n_fft=1024        # number of fft points.
n_shift=256       # number of shift points.
win_length=""     # window length.

# training related
train_config= # config for training.
train_args=   # arguments for training (e.g. "--max_epoch 1").
              # it will overwrite args in train config.
tag=""        # tag for managing experiments.

# decoding related
decode_config= # config for decoding.
decode_args=   # arguments for decoding (e.g. "--threshold 0.75").
               # it will overwrite args in decode config.
decode_tag=""  # tag for decoding directory
decode_model=eval.loss.best.pt # decode model path e.g.,
                               # decode_model=train.loss.best.pt
                               # decode_model=3epoch/model.pt
                               # decode_model=eval.acc.best.pt
                               # decode_model=eval.loss.ave.pt
griffin_lim_iters=4 # the number of iterations of Griffin-Lim.

# [Task dependent] set the datadir name created by local/data.sh.
train_set=      # name of training set.
dev_set=        # name of development set.
eval_sets=      # names of evaluation sets. you can specify multiple items.
srctexts=       # source texts. you can specify multiple items.
nlsyms_txt=     # non-linguistic symbol list (needed if existing).
trans_type=char # transcription type.


log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# check feature type
if [ "${feats_type}" = fbank ]; then
    data_feats="${dumpdir}/fbank"
elif [ "${feats_type}" = stft ]; then
    data_feats="${dumpdir}/stft"
elif [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw"
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)"
    else
        tag=train_default
    fi
    # add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/ //g")"
    fi
fi
if [ -z "${decode_tag}" ]; then
    if [ -n "${decode_config}" ]; then
        decode_tag="$(basename "${decode_config}" .yaml)"
    else
        decode_tag=decode_default
    fi
    # add overwritten arg's info
    if [ -n "${decode_args}" ]; then
        decode_tag+="$(echo "${decode_args}" | sed -e "s/--/\_/g" -e "s/ //g")"
    fi
    decode_tag+="_$(echo "${decode_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${dev_set}, etc."
    # [Task dependent] need to create data.sh for new corpus
    local/data.sh
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # TODO(kamo): Change kaldi-ark to npy or HDF5?
    if [ "${feats_type}" = raw ]; then
        log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
        log "Not yet"
        exit 1

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}/${dset}"
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                "data/${dset}/wav.scp" "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank ] || [ "${feats_type}" = stft ] ; then
        log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}/"

        # Generate the fbank features; by default 80-dimensional fbanks on each frame
        for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
            # 1. Copy datadir
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}/${dset}"

            # 2. Feature extract
            # TODO(kamo): Wrapp (nj->_nj) in make_fbank.sh
            _nj=$((nj<$(<"${data_feats}/${dset}/utt2spk" wc -l)?nj:$(<"${data_feats}/${dset}/utt2spk" wc -l)))
            _opts=
            if [ "${feats_type}" = fbank ] ; then
                _opts+="--fs ${fs} "
                _opts+="--fmax ${fmax} "
                _opts+="--fmin ${fmin} "
                _opts+="--n_mels ${n_mels} "
            fi

            # shellcheck disable=SC2086
            scripts/feats/make_"${feats_type}".sh --cmd "${train_cmd}" --nj "${_nj}" \
                --n_fft "${n_fft}" \
                --n_shift "${n_shift}" \
                --win_length "${win_length}" \
                ${_opts} \
                "${data_feats}/${dset}"

            # 3. Create feats_shape
            scripts/feats/feat_to_shape.sh --nj "${nj}" --cmd "${train_cmd}" \
                "${data_feats}/${dset}/feats.scp" "${data_feats}/${dset}/feats_shape" "${data_feats}/${dset}/log"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
        done

        # compute statistics for global mean-variance normalization
        # TODO(kamo): Parallelize?
        pyscripts/feats/compute-cmvn-stats.py \
            --out-filetype npy \
            scp:"cat ${data_feats}/${train_set}/feats.scp ${data_feats}/${dev_set}/feats.scp |" \
            "${data_feats}/${train_set}/cmvn.npy"
    fi
fi


token_list="data/token_list/${trans_type}/tokens.txt"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Generate character level token_list from ${srctexts}"
    mkdir -p "$(dirname ${token_list})"
    # "nlsyms_txt" should be generated by local/data.sh if need
    if [ -n "${nlsyms_txt}" ]; then
        nlsyms="$(<${nlsyms_txt})"
    else
        nlsyms=
    fi
    echo "<unk>" > "${token_list}"
    if [ -n "${nlsyms}" ]; then
        # shellcheck disable=SC2002
        cat ${srctexts} | pyscripts/text/text2token.py -s 1 -n 1 -l "${nlsyms}" \
            | cut -f 2- -d" " | tr " " "\n" | sort -u \
            | grep -v -e '^\s*$' >> "${token_list}"
    else
        # shellcheck disable=SC2002
        cat ${srctexts} | pyscripts/text/text2token.py -s 1 -n 1 \
            | cut -f 2- -d" " | tr " " "\n" | sort -u \
            | grep -v -e '^\s*$' >> "${token_list}"
    fi

    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
        scripts/text/prepare_token.sh --type "${trans_type}" \
            "${data_feats}/${dset}/text" "${token_list}" "${data_feats}/${dset}"
    done
fi


tts_exp="${expdir}/${tag}"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    _train_dir="${data_feats}/${train_set}"
    _dev_dir="${data_feats}/${dev_set}"
    log "Stage 4: TTS Training: train_set=${_train_dir}, dev_set=${_dev_dir}"

    _opts=
    if [ -n "${train_config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.tts_train --print_config --optimizer adam --encoder_decoder transformer
        _opts+="--config ${train_config} "
    fi

    _feats_type="$(<${_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        _shape=utt2num_samples
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _max_length=80000
        _opts+="--feats_extract_conf fs=${fs} "
    else
        _scp=feats.scp
        _shape=feats_shape
        _type=kaldi_ark
        _max_length=800
        _odim="$(<${_train_dir}/feats_shape head -n1 | cut -d ' ' -f 2 | cut -d',' -f 2)"
        _opts+="--odim=${_odim} "
        _opts+="--normalize_conf stats_file=${_train_dir}/cmvn.npy"
    fi

    # FIXME(kamo): max_length is confusing name. How about fold_length?

    log "TTS training started... log: '${tts_exp}/train.log'"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${tts_exp}"/train.log \
        python3 -m espnet2.bin.tts_train \
            --ngpu "${ngpu}" \
            --token_list "${_train_dir}/tokens.txt" \
            --train_data_path_and_name_and_type "${_train_dir}/token_int,text,text_int" \
            --train_data_path_and_name_and_type "${_train_dir}/${_scp},speech,${_type}" \
            --eval_data_path_and_name_and_type "${_dev_dir}/token_int,text,text_int" \
            --eval_data_path_and_name_and_type "${_dev_dir}/${_scp},speech,${_type}" \
            --train_shape_file "${_train_dir}/token_shape" \
            --train_shape_file "${_train_dir}/${_shape}" \
            --eval_shape_file "${_dev_dir}/token_shape" \
            --eval_shape_file "${_dev_dir}/${_shape}" \
            --resume_epoch latest \
            --max_length 150 \
            --max_length ${_max_length} \
            --output_dir "${tts_exp}" \
            ${_opts} ${train_args}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Decoding: training_dir=${tts_exp}"

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    _opts=
    if [ -n "${decode_config}" ]; then
        _opts+="--config ${decode_config} "
    fi

    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${tts_exp}/${decode_tag}_${dset}"
        _logdir="${_dir}/log"
        mkdir -p "${_logdir}"

        # 0. Copy feats_type
        cp "${_data}/feats_type" "${_dir}/feats_type"
        # 1. Split the key file
        key_file=${_data}/token_int
        split_scps=""
        _nj=$((decode_nj<$(<${key_file} wc -l)?decode_nj:$(<${key_file} wc -l)))
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/tts_decode.*.log'"
        # shellcheck disable=SC2086
        # NOTE(kan-bayashi): --key_file is useful when we want to use multiple data
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts_decode.JOB.log \
            python3 -m espnet2.bin.tts_decode \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/token_int,text,text_int" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --model_file "${tts_exp}"/"${decode_model}" \
                --train_config "${tts_exp}"/config.yaml \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts} ${decode_args}

        # 3. Concatenates the output files from each jobs
         for i in $(seq "${_nj}"); do
              cat "${_logdir}/output.${i}/feats.scp"
         done | LC_ALL=C sort -k1 >"${_dir}/feats.scp"
    done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Synthesis: training_dir=${tts_exp}"
    for dset in "${dev_set}" ${eval_sets}; do
        _dir="${tts_exp}/${decode_tag}_${dset}"

        # TODO(kamo): Wrapp (nj->_nj) in convert_fbank.sh
        _nj=$((nj<$(<"${_dir}/feats.scp" wc -l)?nj:$(<"${_dir}/feats.scp" wc -l)))

        _feats_type="$(<"${_dir}/feats_type")"
        _opts=
        if [ "${_feats_type}" = fbank ] ; then
            _opts+="--n_mels ${n_mels} "
        fi
        # shellcheck disable=SC2086
        scripts/tts/convert_fbank.sh --nj "${_nj}" --cmd "${train_cmd}" \
            --fs "${fs}" \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft "${n_fft}" \
            --n_shift "${n_shift}" \
            --win_length "${win_length}" \
            --iters "${griffin_lim_iters}" \
            ${_opts} \
            "${_dir}" "${_dir}/log" "${_dir}/wav"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
