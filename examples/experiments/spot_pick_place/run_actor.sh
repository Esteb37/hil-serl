# HIL-SERL actor (on the robot): runs the policy; take over with the gamepad whenever it heads somewhere useless.
# Evaluate a checkpoint with --eval_checkpoint_step=N --eval_n_trajs=20.
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=spot_pick_place \
    --checkpoint_path=../../experiments/spot_pick_place/run \
    --actor
