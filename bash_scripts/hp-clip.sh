set -ex

GPUID=7
NAME=clip

# # videov2
# for VAR in 3 5 10 20 30
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${VAR// /_} \
#     --print_freq 1 \
#     --setting videov2 \
#     --clip_num ${VAR}
# done

# # weakly
# for VAR in 3 5 10 20 30
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${VAR// /_} \
#     --print_freq 1 \
#     --setting weakly \
#     --clip_num ${VAR}
# done

# weakly+noisy
for VAR in 3 5 10 20 30
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --num_classes 3 \
    --clip_num ${VAR}
done
