# HIL-SERL actor: runs the policy on the training maps; take over with the gamepad whenever it heads somewhere useless.
# Evaluate a checkpoint on the held-out maps with --eval_checkpoint_step=N --eval_n_trajs=20.
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=car_nav \
    --checkpoint_path=../../experiments/car_nav/run \
    --actor
