set -x

GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTATYPE=${4}
PARTITION=${5}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""} # 
PY_ARGS=${PY_ARGS:-""}
IMAGENET_DIR=${IMAGENET_DIR:-"/mnt/cache/share/images"}

BASENAME=`basename ${0} .sh`
DIR=./exp/pretrain/${BASENAME}
mkdir -p ${DIR}

TOTAL_BATCH_SIZE=4096
let BATCH_SIZE=${TOTAL_BATCH_SIZE}/${GPUS}

EPOCHS=1600
let CKPT_LAST_EPOCH=EPOCHS-1
STOP_SIGN=${DIR}/checkpoint-${CKPT_LAST_EPOCH}.pth
COLLAPSE_SIGN=${DIR}/collapse
AUTO_RERUN_CHECKPOINT=${DIR}/checkpoint-autorerun.pth

while [ ! -f ${STOP_SIGN} ]
do
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
      python -u main_pretrain.py \
        --batch_size ${BATCH_SIZE} \
        --model addp_vit_base_patch16 \
        --input_size 256 \
        --mask_ratio_min 0.5 \
        --mask_ratio_max 1.0 \
        --mask_ratio_mu 0.55 \
        --mask_ratio_std 0.25 \
        --norm_pix_loss \
        --epochs ${EPOCHS} \
        --warmup_epochs 40 \
        --blr 1.0e-4 --weight_decay 0.05 \
        --data_path ${IMAGENET_DIR} \
        --output_dir ${DIR} \
        --log_dir ${DIR} \
        --clip_grad 3.0 \
        --amp_growth_interval 500 \
        --save_latest_freq 5 \
        --token_predictor_ckpt 'exp/pretrained_model/mage-vitb-1600.pth' \
        --decoder_depth 8 \
        ${@:6} ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout_${SUFFIX}.log
done
