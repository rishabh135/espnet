#!/bin/bash
mindepth=0
maxdepth=1

. utils/parse_options.sh

if [ $# -gt 1 ]; then
    echo "Usage: $0 --mindepth 0 --maxdepth 1 [exp]" 1>&2
    echo ""
    echo "Show the system environments and the evaluation results in Markdown format."
    echo 'The default of <exp> is "exp/".'
    exit 1
fi

[ -f ./path.sh ] && . ./path.sh
set -euo pipefail
if [ $# -eq 1 ]; then
    exp=$1
else
    exp=exp
fi


cat << EOF
<!-- Generated by $0 -->
# RESULTS
## Environments
- date: \`$(LC_ALL=C date)\`
EOF

python3 << EOF
import sys, espnet, chainer, torch
pyversion = sys.version.replace('\n', ' ')

print(f"""- python version: \`{pyversion}\`
- espnet version: \`espnet {espnet.__version__}\`
- pytorch version: \`pytorch {torch.__version__}\`""")
EOF

cat << EOF
- Git hash: \`$(git rev-parse HEAD)\`
  - Commit date: \`$(git log -1 --format='%cd')\`

EOF

while IFS= read -r expdir; do
    if ls "${expdir}"/inference_*/score_wer/result.txt &> /dev/null; then
        echo "## $(basename ${expdir})"
        for type in wer cer ter; do
            if ls "${expdir}"/inference_*/score_${type}/result.txt &> /dev/null; then
                cat << EOF
### ${type^^}

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
EOF
                grep -e Avg "${expdir}"/inference_*/score_${type}/result.txt \
                    | sed -e "s#${expdir}/\([^/]*\)/score_${type}/result.txt:#|\1#g" \
                    | sed -e 's#Sum/Avg##g' | tr '|' ' ' | tr -s ' ' '|'
                echo
            fi
        done
    fi


done < <(find ${exp} -mindepth ${mindepth} -maxdepth ${maxdepth} -type d)
