set -ex

GPUID=4
NAME=lrdecay_cv

# # videov2
# for DECAY in 20 50 100
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${DECAY} \
#     --print_freq 1 \
#     --setting videov2 \
#     --lr_step_size ${DECAY}
# done

# # weakly
# for DECAY in 20 50 100
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_weakly_${DECAY} \
#     --print_freq 1 \
#     --setting weakly \
#     --lr_step_size ${DECAY}
# done

# weakly+noisy
for DECAY in 1 2 5
do
	CUDA_VISIBLE_DEVICES=${GPUID} python cv_manual_train.py \
    --name hp-${NAME}_weakly+noisy_${DECAY} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --num_classes 3 \
    --num_epochs 10 \
    --lr_step_size ${DECAY}
done
