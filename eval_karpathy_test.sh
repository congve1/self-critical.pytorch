#! /usr/bin/bash

id=$1
iter=$2

if [ x$3 != x ]; then
  GPU=$3
else
  GPU=0
fi

folder="karpathy_test_results"
if [ ! -d $folder ]; then
  echo "folder: ${folder} not exists, creating one."
  mkdir $folder
fi

export CUDA_VISIBLE_DEVICES=$GPU
commands=(
'python eval.py --dump_images 0 --num_images 5000 --language_eval 1 --model log_${id}/model-${iter}.pth --infos_path log_${id}/infos_${id}-${iter}.pkl --beam_size 1 >karpathy_test_results/${id}_model_${iter}.txt 2>&1'
'python eval.py --dump_images 0 --num_images 5000 --language_eval 1 --model log_${id}/model-${iter}.pth --infos_path log_${id}/infos_${id}-${iter}.pkl --beam_size 2 >>karpathy_test_results/${id}_model_${iter}.txt 2>&1'
'python eval.py --dump_images 0 --num_images 5000 --language_eval 1 --model log_${id}/model-${iter}.pth --infos_path log_${id}/infos_${id}-${iter}.pkl --beam_size 3 >>karpathy_test_results/${id}_model_${iter}.txt 2>&1'
'python eval.py --dump_images 0 --num_images 5000 --language_eval 1 --model log_${id}/model-${iter}.pth --infos_path log_${id}/infos_${id}-${iter}.pkl --beam_size 4 >>karpathy_test_results/${id}_model_${iter}.txt 2>&1'
'python eval.py --dump_images 0 --num_images 5000 --language_eval 1 --model log_${id}/model-${iter}.pth --infos_path log_${id}/infos_${id}-${iter}.pkl --beam_size 5 >>karpathy_test_results/${id}_model_${iter}.txt 2>&1'
)
for command in "${commands[@]}"; do
    while true; do
        echo "execute ${command}"
        eval "${command}" && break
    done
done
unset CUDA_VISIBLE_DEVICES
grep -E '(model_|beam_size|bad)' karpathy_test_results/${id}_model_${iter}.txt
