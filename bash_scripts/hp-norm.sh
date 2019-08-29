set -ex

GPUID=5
NAME=norm

# videov2
for VAR in norm none
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR} \
    --print_freq 1 \
    --setting videov2 \
    --norm ${VAR}
done

# weakly
for VAR in norm none
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly_${VAR} \
    --print_freq 1 \
    --setting weakly \
    --norm ${VAR}
done

# weakly+noisy
for VAR in norm none
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly+noisy_${VAR} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --norm ${VAR}
done
