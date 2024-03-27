set -x

GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTATYPE=${4}
PARTITION=${5}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
CKPT=${6}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}
IMAGENET_DIR=${IMAGENET_DIR:-"/mnt/cache/share/images"}

if [ $GPUS -lt 8 ]; then
    NODE=1
else
    NODE=$[GPUS/GPUS_PER_NODE]
fi

TOTAL_BATCH_SIZE=1024
let BATCH_SIZE=${TOTAL_BATCH_SIZE}/${GPUS}

SUFFIX=`date '+%Y%m%d%H%M%S'`
BASENAME=`basename ${CKPT} .pth`
DIR=./exp/finetune/${BASENAME}
mkdir -p ${DIR}

srun --partition=${PARTITION} \
    ${SRUN_ARGS} \
    --quotatype=${QUOTATYPE} \
    --nodes=${NODE} \
    --job-name=${JOB_NAME} \
    -n$GPUS \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    python -u main_finetune.py \
    --output_dir ${DIR} \
    --log_dir ${DIR} \
    --batch_size ${BATCH_SIZE} \
    --model vit_large_patch16 \
    --input_size 256 \
    --finetune ${CKPT} \
    --epochs 50 \
    --blr 2.5e-4 --layer_decay 0.8 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --warmup_epochs 5 \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout_${SUFFIX}.log

