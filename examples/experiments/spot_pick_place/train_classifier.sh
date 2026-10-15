# classifier_data/ -> classifier_ckpt/
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_reward_classifier.py "$@" \
    --exp_name=spot_pick_place
