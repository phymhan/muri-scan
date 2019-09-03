set -ex

GPUID=6
NAME=norm

# videov2
for VAR in 'norm --norm_fc norm' 'none --norm_fc norm' 'none --norm_fc none' 'norm --norm_fc none'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting videov2 \
    --dim_fc 64 \
    --norm_input_map ${VAR}
done

# weakly
for VAR in 'norm --norm_fc norm' 'none --norm_fc norm' 'none --norm_fc none' 'norm --norm_fc none'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --dim_fc 64 \
    --norm_input_map ${VAR}
done

# weakly+noisy
for VAR in 'norm --norm_fc norm' 'none --norm_fc norm' 'none --norm_fc none' 'norm --norm_fc none'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly+noisy_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --dim_fc 64 \
    --num_classes 3 \
    --norm_input_map ${VAR}
done
