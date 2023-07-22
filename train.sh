seed=12
python main.py \
    --base_folder exp-mnli-poe \
    --seed $seed \
    --data mnli \
    --loss poe \
    --cuda 0 \
    --debug 0

