#!/usr/bin/env bash
export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`

OUT_DIR_PREFIX=$CACHE_DIR

SPLIT=test
GPU=${1:-0}
SOURCE=${2:-java};
TARGET=${3:-python};
DATA_SRC=${4:-new_g4g};
TRAIN_BATCH_SIZE=${5:-4} # per_gpu_train_bsz * num_gpu
NUM_TRAIN_EPOCHS_OR_UPDATES=${6:-20}
LR=${7:-1e-4}
EVAL_BATCH_SIZE=${8:-8}
GRAD_ACCUM_STEP=${9:-8}
MODEL=${10:-codet5_base}
SEED=${11:-1234}


HYP_PATH_PREFIX=${SAVE_DIR_HOME}/${DATA_SRC}/${MODEL}/${SOURCE}2${TARGET}_bs${TRAIN_BATCH_SIZE}_${GRAD_ACCUM_STEP}_epoch${NUM_TRAIN_EPOCHS_OR_UPDATES}_lr${LR}
if [ $SEED != 1234 ]; then
    HYP_PATH_PREFIX=${HYP_PATH_PREFIX}_seed${SEED}
    echo $HYP_PATH_PREFIX
fi

DATA_DIR=${CODE_DIR_HOME}/data/transcoder_${SOURCE}2${TARGET}

module load gcc-7.4

if [[ $SOURCE == 'cpp' ]]; then
    if [[ $TARGET == 'java' ]]; then
        lang_pair_name=cpp-java 
    elif [[ $TARGET == 'python' ]]; then
        lang_pair_name=cpp-python 
    fi
elif [[ $SOURCE == 'java' ]]; then
    if [[ $TARGET == 'cpp' ]]; then
        lang_pair_name=cpp-java 
    elif [[ $TARGET == 'python' ]]; then
        lang_pair_name=java-python 
    fi
elif [[ $SOURCE == 'python' ]]; then
    if [[ $TARGET == 'cpp' ]]; then
        lang_pair_name=cpp-python 
    elif [[ $TARGET == 'java' ]]; then
        lang_pair_name=java-python 
    fi
fi

echo Evaluating language pair: $lang_pair_name

function run_transcoder_eval_multi_beam () {
NUM_RETURN_SEQ=1
BEAM_NUM=10
TEMP=0.8

for (( i=0; i<${NUM_RETURN_SEQ}; i++ ))
do
    HYP_PATH=${HYP_PATH_PREFIX}/transcoder_eval_ppl_multi_output_beam_${TEMP}_${BEAM_NUM}/${SPLIT}.output${i};
    OUTDIR=${OUT_DIR_PREFIX}/${HYP_PATH_PREFIX}/transcoder_eval_ppl_multi_output_beam_${TEMP}_${BEAM_NUM}/test_scripts${i}
    mkdir -p $OUTDIR

    echo "Process the result in $HYP_PATH"

    export PYTHONPATH=$CODE_DIR_HOME;
    python compute_ca.py \
        --src_path ${DATA_DIR}/${SPLIT}.${lang_pair_name}.${SOURCE} \
        --ref_path ${DATA_DIR}/${SPLIT}.${lang_pair_name}.${TARGET} \
        --hyp_paths $HYP_PATH \
        --id_path ${DATA_DIR}/${SPLIT}.${lang_pair_name}.id \
        --split $SPLIT \
        --outfolder $OUTDIR \
        --source_lang $SOURCE \
        --target_lang $TARGET \
        --out_prefix ${SOURCE}-${TARGET}.${SPLIT}.$i \
        --retry_mismatching_types True;

    rm $OUTDIR/*
done
}

function run_train_multi_beam () {
DATA_DIR=${CODE_DIR_HOME}/data/${DATA_SRC}_train

NUM_RETURN_SEQ=50
TEMP=0.8

for (( i=0; i<${NUM_RETURN_SEQ}; i++ ))
do
   echo "Process the result of candidate $i"
    HYP_PATH=${HYP_PATH_PREFIX}/train_ppl_multi_output_beam_${TEMP}/${SPLIT}.output${i};
    OUTDIR=${OUT_DIR_PREFIX}/${HYP_PATH_PREFIX}/train_ppl_multi_output_beam_${TEMP}/test_scripts${i}
    mkdir -p $OUTDIR

    export PYTHONPATH=$CODE_DIR_HOME;
    python compute_ca.py \
        --src_path ${DATA_DIR}/${SPLIT}.${lang_pair_name}.${SOURCE} \
        --ref_path ${DATA_DIR}/${SPLIT}.${lang_pair_name}.${TARGET} \
        --hyp_paths $HYP_PATH \
        --id_path ${DATA_DIR}/${SPLIT}.${lang_pair_name}.id \
        --split $SPLIT \
        --outfolder $OUTDIR \
        --source_lang $SOURCE \
        --target_lang $TARGET \
        --out_prefix ${SOURCE}-${TARGET}.${SPLIT}.$i \
        --retry_mismatching_types True;

    rm $OUTDIR/*
done
}

run_transcoder_eval_multi_beam;

# run_train_multi_beam;
