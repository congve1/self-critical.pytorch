#! /usr/bin/bash

if [ x$1 != x ]; then
  id=$1
else
  echo "Error: no id specified"
  exit
fi

if [ x$2 != x ]; then
  iter=$2
else
  echo "Error: no iter sepcified"
  exit
fi

model_path=log_$id/model-$iter.pth
infos_path=log_$id/infos_$id-$iter.pkl
echo "model path: $model_path"
echo "infos_path: $infos_path"

if [ x$3 != x ]; then
  beam_size=$3
else
  beam_size=1
fi

echo "beam_size: $beam_size"

if [ x$4 != x ]; then
  GPU=$4
else
  GPU=0
fi

echo "CUDA_VISIBLE_DEVICES=$GPU"

export CUDA_VISIBLE_DEVICES=$GPU
python generate_online_results.py --dump_images 0 --alg_name ${id}${iter} --model $model_path --infos_path $infos_path --beam_size $beam_size --batch_size 1000
unset CUDA_VISIBLE_DEVICES
