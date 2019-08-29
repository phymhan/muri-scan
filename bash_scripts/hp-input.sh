set -ex

GPUID=6
NAME=input

# videov2
for VAR in '128' '64 128' '128 128' '128 64 128' '128 32 128' '128 64'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting videov2 \
    --dim_input_map ${VAR}
done

# weakly
for VAR in '128' '64 128' '128 128' '128 64 128' '128 32 128' '128 64'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --dim_input_map ${VAR}
done

# weakly+noisy
for VAR in '128' '64 128' '128 128' '128 64 128' '128 32 128' '128 64'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --dim_input_map ${VAR}
done
