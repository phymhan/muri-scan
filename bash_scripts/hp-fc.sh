set -ex

GPUID=7
NAME=fc

# # videov2
# for VAR in '32' '64' '128' '32 32' '64 64' '128 128'
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${VAR// /_} \
#     --print_freq 1 \
#     --setting videov2 \
#     --dim_fc ${VAR}
# done

# # weakly
# for VAR in '32' '64' '128' '32 32' '64 64' '128 128'
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${VAR// /_} \
#     --print_freq 1 \
#     --setting weakly \
#     --dim_fc ${VAR}
# done

# weakly+noisy
for VAR in '32' '64' '128' '32 32' '64 64' '128 128'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --num_classes 3 \
    --dim_fc ${VAR}
done
