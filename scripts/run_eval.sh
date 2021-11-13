set -x
CKPT_PATH=$1
CONFIG_PATH=$2

export CODE_TRANSFORMER_DATA_PATH=/scratch1/08401/ywen/data/code-transformer
export CODE_TRANSFORMER_BINARY_PATH=/scratch1/08401/ywen/code/code-transformer/sub_modules
export CODE_TRANSFORMER_MODELS_PATH=/scratch1/08401/ywen/code/code-transformer/models
export CODE_TRANSFORMER_LOGS_PATH=/scratch1/08401/ywen/code/code-transformer/logs
export CODE_TRANSFORMER_DATA_PATH_STAGE_2=/scratch1/08401/ywen/data/code-transformer/stage2

#CUDA_VISIBLE_DEVICES=0 python -m scripts.run-experiment code_transformer/experiments/code_transformer/code_summarization.yaml
#CUDA_VISIBLE_DEVICES=0 python -m scripts.run-experiment code_transformer/experiments/paper/ct_multilang.yaml 
#python -m scripts.run-experiment code_transformer/experiments/paper/ct_multilang.yaml 
echo ${CKPT_PATH}
python -m scripts.evaluate-multilanguage code_transformer 0 96000 test --ckpt-path ${CKPT_PATH} --config ${CONFIG_PATH} 
#python -m scripts.evaluate-multilanguage code_transformer 0 96000 test --ckpt-path ${CKPT_PATH} --config ${CONFIG_PATH} 
#python test.py
