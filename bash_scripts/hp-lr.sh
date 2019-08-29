set -ex

GPUID=4
NAME=lr

# videov2
for LR in 0.01 0.001 0.0002
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${LR} \
    --print_freq 1 \
    --setting videov2 \
    --lr ${LR}
done

# weakly
for LR in 0.01 0.001 0.0002
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly_${LR} \
    --print_freq 1 \
    --setting weakly \
    --lr ${LR}
done

# weakly+noisy
for LR in 0.01 0.001 0.0002
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly+noisy_${LR} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --lr ${LR}
done
