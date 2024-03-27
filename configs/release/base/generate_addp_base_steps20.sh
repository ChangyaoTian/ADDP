set -x

GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTATYPE=${4}
PARTITION=${5}
PRETRAIN=${6}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}
IMAGENET_DIR=${IMAGENET_DIR:-"/mnt/cache/share/images"}

BASENAME=`basename ${0} .sh`
DIR=./exp/generate/${BASENAME}
mkdir -p ${DIR}

TOTAL_BATCH_SIZE=64
let BATCH_SIZE=${TOTAL_BATCH_SIZE}/${GPUS}

SUFFIX=`date '+%Y%m%d%H%M%S'`
srun --partition=${PARTITION} \
  --mpi=pmi2 \
  --quotatype=${QUOTATYPE} \
  --job-name=${JOB_NAME} \
  -n$GPUS \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=$CPUS_PER_TASK \
  --kill-on-bad-exit=1 \
  ${SRUN_ARGS} \
  python -u main_generate.py \
    --batch_size ${BATCH_SIZE} \
    --model addp_vit_base_patch16 \
    --input_size 256 \
    --norm_pix_loss \
    --mask_ratio 1.00 \
    --mask_ratio_min 0.5 \
    --mask_ratio_max 1.0 \
    --mask_ratio_mu 0.55 \
    --mask_ratio_std 0.25 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${DIR} \
    --log_dir ${DIR} \
    --pretrain ${PRETRAIN} \
    --save_dir_suffix '_cosine' \
    --num_iteration 20 \
    --token_predictor_ckpt 'exp/pretrained_model/mage-vitb-1600.pth' \
    --decoder_depth 8 \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout_${SUFFIX}.log
