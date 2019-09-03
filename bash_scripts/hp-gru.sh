set -ex

GPUID=6
NAME=gru

# videov2
for VAR in 'true --gru_hidden_dim 32 --gru_out_dim 32' 'true --gru_hidden_dim 64 --gru_out_dim 64' 'true --gru_hidden_dim 128 --gru_out_dim 128' 'true --gru_hidden_dim 64 --gru_out_dim 32' 'true --gru_hidden_dim 128 --gru_out_dim 64'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_video_${VAR// /_} \
    --print_freq 1 \
    --setting videov2 \
    --use_gru ${VAR}
done

# weakly
for VAR in 'true --gru_hidden_dim 32 --gru_out_dim 32' 'true --gru_hidden_dim 64 --gru_out_dim 64' 'true --gru_hidden_dim 128 --gru_out_dim 128' 'true --gru_hidden_dim 64 --gru_out_dim 32' 'true --gru_hidden_dim 128 --gru_out_dim 64'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --use_gru ${VAR}
done

# weakly+noisy
for VAR in 'true --gru_hidden_dim 32 --gru_out_dim 32' 'true --gru_hidden_dim 64 --gru_out_dim 64' 'true --gru_hidden_dim 128 --gru_out_dim 128' 'true --gru_hidden_dim 64 --gru_out_dim 32' 'true --gru_hidden_dim 128 --gru_out_dim 64'
do
	CUDA_VISIBLE_DEVICES=${GPUID} python main.py \
    --name hp-${NAME}_weakly+noisy_${VAR// /_} \
    --print_freq 1 \
    --setting weakly \
    --noisy true \
    --num_classes 3 \
    --use_gru ${VAR}
done
