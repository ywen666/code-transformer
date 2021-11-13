set -x
CONFIG_PATH=$1
echo ${CONFIG_PATH}

export CODE_TRANSFORMER_DATA_PATH=/scratch1/08401/ywen/data/code-transformer
export CODE_TRANSFORMER_BINARY_PATH=/scratch1/08401/ywen/code/code-transformer/sub_modules
export CODE_TRANSFORMER_MODELS_PATH=/scratch1/08401/ywen/code/code-transformer/models
export CODE_TRANSFORMER_LOGS_PATH=/scratch1/08401/ywen/code/code-transformer/logs
export CODE_TRANSFORMER_DATA_PATH_STAGE_2=/scratch1/08401/ywen/data/code-transformer/stage2

#CUDA_VISIBLE_DEVICES=0 python -m scripts.run-experiment code_transformer/experiments/code_transformer/code_summarization.yaml
#CUDA_VISIBLE_DEVICES=0 python -m scripts.run-experiment code_transformer/experiments/paper/ct_multilang.yaml 
#python -m scripts.run-experiment code_transformer/experiments/paper/ct_multilang_rank1.yaml 
python -m scripts.run-experiment ${CONFIG_PATH} 
#CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate-multilanguage code_transformer 0 96000 test 
#python test.py
