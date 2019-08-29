set -ex

GPUID=5
NAME=dropout

# # videov2
# for p in 0.01 0.1 0.2 0.5 0.8
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${p} \
#     --print_freq 1 \
#     --setting videov2 \
#     --dropout ${p}
# done

# # weakly
# for p in 0.01 0.1 0.2 0.5 0.8
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_weakly_${p} \
#     --print_freq 1 \
#     --setting weakly \
#     --dropout ${p}
# done

# weakly+noisy
for p in 0.01 0.1 0.2 0.5 0.8
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly+noisy_${p} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --num_classes 3 \
    --dropout ${p}
done
