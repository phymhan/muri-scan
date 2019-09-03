set -ex

GPUID=7
NAME=time

# videov2
for VAR in '30 --time_step 2' '24 --time_step 5' '120 --time_step 1'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting videov2 \
    --time_len ${VAR}
done

# weakly
for VAR in '30 --time_step 2' '24 --time_step 5' '120 --time_step 1'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --time_len ${VAR}
done

# weakly+noisy
for VAR in '30 --time_step 2' '24 --time_step 5' '120 --time_step 1'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly+noisy_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --num_classes 3 \
    --time_len ${VAR}
done
