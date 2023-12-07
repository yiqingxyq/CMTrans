#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

# set DIRs
CURRENT_DIR=`pwd`
export PYTHONPATH=$CODE_DIR_HOME;

evaluator_script="${CODE_DIR_HOME}/evaluation";
codebleu_path="${CODE_DIR_HOME}/evaluation/CodeBLEU";

# inputs
GPU=${1:-0};
SOURCE=${2:-java};
TARGET=${3:-python};
DATA_SRC=${4:-g4g};
TRAIN_BATCH_SIZE=${5:-4} # per_gpu_train_bsz * num_gpu
NUM_TRAIN_EPOCHS=${6:-20}
LR=${7:-1e-4}
EVAL_BATCH_SIZE=${8:-8}
GRAD_ACCUM_STEP=${9:-8}
BASE_MODEL=${10:-codet5_base} # the name of pretrained model or the name of comparable corpora
SEED=${11:-1234}


export CUDA_VISIBLE_DEVICES=$GPU
echo "Source: $SOURCE Target: $TARGET"

path_2_data=${CODE_DIR_HOME}/data/${DATA_SRC};
echo $path_2_data

SAVE_DIR=${SAVE_DIR_HOME}/${DATA_SRC}/${BASE_MODEL}/${SOURCE}2${TARGET}_bs${TRAIN_BATCH_SIZE}_${GRAD_ACCUM_STEP}_epoch${NUM_TRAIN_EPOCHS}_lr${LR}
if [ $SEED != 1234 ]; then
    SAVE_DIR=${SAVE_DIR}_seed${SEED}
    echo $SAVE_DIR
fi

mkdir -p $SAVE_DIR
mkdir -p $CACHE_DIR

tokenizer_path=${CURRENT_DIR}/tokenizer;
source_length=510;
target_length=510;


function train () {

echo "Running training"

lr=${LR};

if [[ $BASE_MODEL == 'codet5_base' ]]; then
    pretrained_model=Salesforce/codet5-base;
else 
    # train on comparable corpora first 
    pretrained_model=${SAVE_DIR_HOME}/${BASE_MODEL}/codet5_base/${SOURCE}2${TARGET}_bs4_8_epoch20_lr1e-4/checkpoint-best-ppl;
fi
echo TRAINING FROM $pretrained_model
echo SAVE_DIR: $SAVE_DIR

# do not cache training data
mkdir -p $CACHE_DIR/train
rm -r $CACHE_DIR/train/*

python run_gen.py \
    --do_train \
    --do_eval \
    --save_last_checkpoints \
    --always_save_model \
    --task translate \
    --sub_task "${SOURCE}-${TARGET}" \
    --model_type codet5 \
    --tokenizer_name roberta-base \
    --tokenizer_path $tokenizer_path \
    --model_name_or_path $pretrained_model \
    --output_dir $SAVE_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --warmup_steps 100 \
    --learning_rate $lr \
    --patience 5 \
    --data_dir $path_2_data \
    --cache_path $CACHE_DIR/train \
    --res_dir $SAVE_DIR \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEP \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size 5 \
    --seed $SEED 

# copy config
if [ -f "$SAVE_DIR/checkpoint-best-ppl/config.json" ]; then
    echo "config exists"
else 
    cp $tokenizer_path/config.json $SAVE_DIR/checkpoint-best-ppl/config.json
    cp $tokenizer_path/config.json $SAVE_DIR/checkpoint-last/config.json
fi

}


function predict_transcoder_eval_multi_beam () {

echo "Running transcoder eval (multiple candidates, beam sample)"

NUM_RETURN_SEQ=10
BEAM_SIZE=10
TEMP=0.8

DATA_DIR=${CODE_DIR_HOME}/data/transcoder_${SOURCE}2${TARGET};
OUT_DIR=${SAVE_DIR}/transcoder_eval_ppl_multi_output_beam_${TEMP}_${BEAM_SIZE};
mkdir -p $OUT_DIR;
RESULT_FILE=${OUT_DIR}/result;


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

GROUND_TRUTH_PATH=${DATA_DIR}/test.${lang_pair_name}.${TARGET};

MODEL_PATH=${SAVE_DIR}/checkpoint-best-ppl # use a finetuned ckpt 

mkdir -p $CACHE_DIR/transcoder_eval_ppl_multi_output_beam_${TEMP}_${BEAM_SIZE}
rm -r $CACHE_DIR/transcoder_eval_ppl_multi_output_beam_${TEMP}_${BEAM_SIZE}/*

python run_gen.py \
    --do_test \
    --model_type codet5 \
    --config_name Salesforce/codet5-base \
    --tokenizer_name roberta-base \
    --tokenizer_path $tokenizer_path \
    --model_name_or_path $MODEL_PATH \
    --task translate \
    --sub_task "${SOURCE}-${TARGET}" \
    --output_dir $SAVE_DIR \
    --data_dir $DATA_DIR \
    --cache_path $CACHE_DIR/transcoder_eval_ppl_multi_output_beam_${TEMP}_${BEAM_SIZE} \
    --res_dir $OUT_DIR \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $BEAM_SIZE \
    --num_return_sequences $NUM_RETURN_SEQ \
    --temp $TEMP \
    2>&1 | tee ${OUT_DIR}/evaluation.log;

python $evaluator_script/evaluator.py \
    --references $GROUND_TRUTH_PATH \
    --txt_ref \
    --predictions $OUT_DIR/test.output0 \
    --language $TARGET \
    2>&1 | tee $RESULT_FILE;


cd $codebleu_path;
python calc_code_bleu.py \
    --ref $GROUND_TRUTH_PATH \
    --txt_ref \
    --hyp $OUT_DIR/test.output0 \
    --lang $TARGET \
    --params "1,0,1,1" \
    2>&1 | tee -a $RESULT_FILE;

cd $CURRENT_DIR;

python $evaluator_script/compile.py \
    --input_file $OUT_DIR/test.output0 \
    --language $TARGET \
    2>&1 | tee -a $RESULT_FILE;

count=`ls -1 *.class 2>/dev/null | wc -l`;
[[ $count != 0 ]] && rm *.class;

}

function predict_train_multi_beam () {
    
echo "Generate output for train set (multiple candidates, greedy beam)"

NUM_RETURN_SEQ=20
BEAM_SIZE=20
TEMP=0.8


DATA_DIR=${path_2_data}_train
OUT_DIR=${SAVE_DIR}/train_ppl_multi_output_beam_${TEMP};


mkdir -p $OUT_DIR;

MODEL_PATH=${SAVE_DIR}/checkpoint-best-ppl # use my finetuned ckpt 

mkdir -p $CACHE_DIR/train_ppl_multi_output_beam_${TEMP}
rm -r $CACHE_DIR/train_ppl_multi_output_beam_${TEMP}/*

python run_gen.py \
    --do_test \
    --model_type codet5 \
    --config_name Salesforce/codet5-base \
    --tokenizer_name roberta-base \
    --tokenizer_path $tokenizer_path \
    --model_name_or_path $MODEL_PATH \
    --task translate \
    --sub_task "${SOURCE}-${TARGET}" \
    --output_dir $SAVE_DIR \
    --data_dir $DATA_DIR \
    --cache_path $CACHE_DIR/train_ppl_multi_output_beam_${TEMP} \
    --res_dir $OUT_DIR \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $BEAM_SIZE \
    --num_return_sequences $NUM_RETURN_SEQ \
    --temp $TEMP \
    2>&1 | tee ${OUT_DIR}/evaluation.log;
}


train;
predict_transcoder_eval_multi_beam;
