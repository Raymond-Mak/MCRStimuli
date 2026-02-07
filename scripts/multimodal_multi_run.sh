#!/usr/bin/env bash
set -euo pipefail

# 设置环境变量
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/root/autodl-tmp/LaFTer-masterTEXT"
source /etc/network_turbo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 默认参数
DEFAULT_VIT_LR=1e-6
DEFAULT_BERT_LR=5e-6
DEFAULT_HEAD_LR=5e-4

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

resolve_img_root() {
    local dataset=$1
    case "${dataset}" in
        Twitter1|Twitter2|FI|Emoset) printf "data/%s\n" "${dataset}" ;;
        Emotion6|FI_new|FI_Probing) printf "data\n" ;;
        *) printf "data/%s\n" "${dataset}" ;;
    esac
}

# 读取运行配置 - 这里可以添加更多配置
RUNS=(
    "FI_Probing|caption/FI_Probing/gpt_narracap_extended_FI_Probing.csv|ViT-B/32|32|50|42|0|0|1e-4|--freeze_vision --freeze_bert"
    "FI_Probing|caption/FI_Probing/gpt_narracap_extended_FI_Probing.csv|ViT-B/32|32|10|42|1e-6|0|5e-4|--freeze_bert"
    "FI_Probing|caption/FI_Probing/gpt_narracap_extended_FI_Probing.csv|ViT-B/32|32|10|42|0|5e-6|5e-4|--freeze_vision"
)

total_runs=${#RUNS[@]}
log_msg "开始运行 ${total_runs} 个实验"

for ((i=0; i<total_runs; i++)); do
    cfg="${RUNS[$i]}"
    run_idx=$((i+1))

    # 解析配置
    IFS='|' read -r dataset caption arch batch epochs seed vit_lr bert_lr head_lr extra_flags <<< "${cfg}"

    # 设置默认值
    [[ -z "${arch}" ]] && arch="ViT-B/32"
    [[ -z "${batch}" ]] && batch=32
    [[ -z "${epochs}" ]] && epochs=20
    [[ -z "${seed}" ]] && seed=7777
    [[ -z "${vit_lr}" || "${vit_lr}" == "default" ]] && vit_lr=${DEFAULT_VIT_LR}
    [[ -z "${bert_lr}" || "${bert_lr}" == "default" ]] && bert_lr=${DEFAULT_BERT_LR}
    [[ -z "${head_lr}" || "${head_lr}" == "default" ]] && head_lr=${DEFAULT_HEAD_LR}

    # 创建日志目录
    log_dir="${REPO_ROOT}/log/multimodal_new/${dataset}"
    mkdir -p "${log_dir}"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${log_dir}/multimodal_${dataset}_${timestamp}.log"

    log_msg "(${run_idx}/${total_runs}) 开始处理数据集: ${dataset}"
    log_msg "配置: arch=${arch}, batch=${batch}, epochs=${epochs}, seed=${seed}"

    # 设置图片路径
    img_root=$(resolve_img_root "${dataset}")
    out_dir="${REPO_ROOT}/cache"
    out_prefix="${dataset}"

    export PYTHONHASHSEED=${seed}

    # 第一步：构建数据对
    log_msg "构建Dassl数据对..."
    /root/miniconda3/envs/LaFTer/bin/python "${REPO_ROOT}/build_pairs_dassl.py" \
        --dataset "${dataset}" \
        --img_root "${img_root}" \
        --caption_file "${caption}" \
        --out_dir "${out_dir}" \
        --out_prefix "${out_prefix}" 2>&1 | tee -a "${log_file}"

    # 获取生成的数据文件路径
    cap_stem=$(basename "${caption}")
    cap_stem=${cap_stem%.*}
    train_pairs="${out_dir}/${out_prefix}_${dataset}_${cap_stem}_train.jsonl"
    val_pairs="${out_dir}/${out_prefix}_${dataset}_${cap_stem}_val.jsonl"

    # 第二步：多模态训练
    log_msg "开始多模态训练..."
    /root/miniconda3/envs/LaFTer/bin/python "${REPO_ROOT}/LaFTer.py" \
        --pipeline multimodal \
        --root "${img_root}" \
        --arch "${arch}" \
        --encoder-type clip \
        --seed "${seed}" \
        --train_pairs "${train_pairs}" \
        --val_pairs "${val_pairs}" \
        --mm_batch_size "${batch}" \
        --mm_epochs "${epochs}" \
        --vit_lr "${vit_lr}" \
        --bert_lr "${bert_lr}" \
        --head_lr "${head_lr}" \
        --fusion_dim 768 \
        --fusion_type standard \
        --enable_wandb \
        --mm_save_dir "checkpoints/mm_${dataset}" \
        ${extra_flags} 2>&1 | tee -a "${log_file}"

    log_msg "实验 ${run_idx} 完成，日志保存至: ${log_file}"
done

log_msg "所有 ${total_runs} 个实验运行完成！"