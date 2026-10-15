# BC baseline on demo_data/*.pkl -> bc_ckpt/. Evaluate on the robot with --eval_n_trajs=10.
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_bc.py "$@" \
    --exp_name=spot_pick_place \
    --bc_checkpoint_path=../../experiments/spot_pick_place/bc_ckpt
