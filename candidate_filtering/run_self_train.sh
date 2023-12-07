CKPT=$1 # new_g4g/codet5_base/java2python_bs4_8_epoch20_lr1e-4
TEST_CASE_NUM=${2:-10};
CANDIDATE_END=${3:-20}
GPU=${4:-0};

CANDIDATE_START=0

# ckpt specific info 
CKPT_DIR=${SAVE_DIR_HOME}/${CKPT}
RESULT_DIR=${CKPT_DIR}/train_ppl_multi_output_beam_0.8
TMP_DIR=${CACHE_DIR}/${CKPT}/train_ppl_multi_output_beam_0.8

DATASET_DIR=${CODE_DIR_HOME}/data/new_g4g_functions_train
ID_FILE=${DATASET_DIR}/test.java-python.id
READ_FUNC_FILE=${DATASET_DIR}/test.java-python.java
# READ_FUNC_FILE=${RESULT_DIR}/test.output0
ARG_LANG=java 
SRC_LANG=java
TGT_LANG=python

SRC_FUNC_FILE=${DATASET_DIR}/test.java-python.${SRC_LANG}

# get func_subsets
CUDA_VISIBLE_DEVICES=$GPU python split_samples.py --id_file $ID_FILE --result_dir $RESULT_DIR \
                                    --subset_folder func_subsets --split_num 30

# generate inputs
for i in {0..29}
do 
    CUDA_VISIBLE_DEVICES=$GPU python generate_inputs.py --subset_idx $i \
        --result_dir $RESULT_DIR --id_file $ID_FILE --func_file $READ_FUNC_FILE --lang $ARG_LANG \
        --test_case_num $TEST_CASE_NUM --subset_folder func_subsets &
done 

wait

# filter inputs
for i in {0..29}
do 
    CUDA_VISIBLE_DEVICES=$GPU python filter_inputs.py --subset_idx $i --tmp_dir $TMP_DIR \
        --result_dir $RESULT_DIR --id_file $ID_FILE --func_file $SRC_FUNC_FILE --lang $SRC_LANG \
        --subset_folder func_subsets &
done

wait

# filter candidates 
for i in {0..29}
do 
    CUDA_VISIBLE_DEVICES=$GPU python filter_candidates.py --subset_idx $i --tmp_dir $TMP_DIR \
        --candidate_start $CANDIDATE_START --candidate_end $CANDIDATE_END \
        --result_dir $RESULT_DIR --id_file $ID_FILE --lang $TGT_LANG \
        --subset_folder func_subsets &
done 

wait 

echo Finished candidate filtering
