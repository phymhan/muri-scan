set -ex

GPUID=6
NAME=coarse_cv

# # videov2
# for VAR in '128' '64 128' '128 128' '128 64 128' '128 32 128' '128 64'
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${VAR// /_} \
#     --print_freq 1 \
#     --setting videov2 \
#     --dim_input_map ${VAR}
# done

# # weakly
# for VAR in '128' '64 128' '128 128' '128 64 128' '128 32 128' '128 64'
# do
# 	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
#     --name hp-${NAME}_video_${VAR// /_} \
#     --print_freq 1 \
#     --setting weakly \
#     --dim_input_map ${VAR}
# done

# weakly+noisy
for VAR in '128 128'
do
        CUDA_VISIBLE_DEVICES=${GPUID} python cv_manual_train.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --num_classes 3 \
    --dim_input_map 128 128 \
    --lr 0.001 \
    --lr_step_size 50 \
    --dropout 0.5 \
    --dim_fc 64 64 \
    --norm_input_map none \
    --norm_fc none \
    --clip 10 \
    --time_len 120 \
    --time_step 1 
done
