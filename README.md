# CMTrans

- Official code of our EMNLP'23 Findigns paper, [Data Augmentation for Code Translation with Comparable Corpora and Multiple References](https://arxiv.org/abs/2311.00317). 
- We adapt the codebase from the repository of [AVATAR](https://github.com/wasiahmad/AVATAR)



## Table of Contents

- [CMTrans](#CMTrans)
  - [Table of Contents](#table-of-contents)
  - [Environment](#environment)
  - [Training & Evaluation](#training--evaluation)
  - [Citation](#citation)

## Environment
To solve the environment, we recommend you to create a new environment:
```
conda create -n "code_trans_env" python=3.7
source activate code_trans_env
conda config --add channels conda-forge
conda config --add channels pytorch
```

### step 0: set up environment variables
Set up the following variables in `setup.sh`:
`$HOME_DIR`: the directory where you clone this repo into
`$STORAGE_DIR`: the directory to store the models and results
`$CACHE_DIR`: the cache directory

Then run:
```
source setup.sh
```

### Step 1: install packages
```
conda install six scikit-learn stringcase ply slimit astunparse submitit
conda transformers=="3.0.2"
pip install cython
```

You also need to install Pytorch based on your own CUDA version. For example:
```
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
```

### Step 2: make a directory for external libraries
```
mkdir -p third_party
```

### Step 3: install tree-sitter in third_party/
```
cd third_party

git clone https://github.com/tree-sitter/tree-sitter-cpp.git
git clone https://github.com/tree-sitter/tree-sitter-java.git
git clone https://github.com/tree-sitter/tree-sitter-python.git

cd ../
```

### Step 4: install fairseq in third_party/
```
cd third_party

git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout 698e3b91ffa832c286c48035bdff78238b0de8ae
pip install .
cd ../

cd ../
```

### Step 4.5 (Optional): upgrade gcc to 5.0.0+
Before installing apex, you will need to check your gcc version by:
```
gcc --version
```
If your gcc version is lower than 5.0.0, you need to upgrade it to 5.0.0 or a higher version:
```
sudo apt-get update
sudo apt install build-essential

```
If you do not have root access, you can first check if there are other gcc versions in your system:
```
module avail
```

If there are some gcc available, you can simply load it by:
```
module load gcc-X.X.X
```

If there are not, you will need to install gcc by source.

### Step 5: install apex in third_party/
After upgrading gcc to 5.0.0+, run the following command to install Apex:
```
cd third_party

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../

cd ../
```

### Step 6: install the following packages
```
pip install sacrebleu=="1.2.11" javalang tree_sitter==0.19.0 psutil fastBPE setuptools==59.5.0 sentencepiece
```

### Step 7: build library 
To build the library, you need to first quit your conda environment. Check [here](https://github.com/explosion/spaCy/issues/2447#issuecomment-400102779) for details.
```
conda deactivate 
python build.py
```


## Training & Evaluation

To train and evaluate a model, go to the corresponding model directory and execute the **codet5/run.sh** script.

To finetune and evaluate CodeT5:
```
# train
cd codet5
bash run.sh

# evaluation 
cd evaluation 
bash run.sh
```

## Citation

```
@inproceedings{xie2023data,
    title = {Data Augmentation for Code Translation with Comparable Corpora and Multiple References},
    author = {Yiqing Xie and Atharva Naik and Daniel Fried and Carolyn Rose},
    year = {2023},
    selected={true},
    booktitle = {Findings of the 2023 Conference on Empirical Methods in Natural Language Processing: EMNLP 2023},
    pdf={https://arxiv.org/pdf/2311.00317.pdf}
}
```
