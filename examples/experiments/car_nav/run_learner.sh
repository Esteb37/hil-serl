# HIL-SERL learner (no window): demo_data/ + online data from the actor. Restarting with the same checkpoint_path resumes.
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=car_nav \
    --checkpoint_path=../../experiments/car_nav/run \
    $(for f in demo_data/*.pkl; do echo --demo_path=$f; done) \
    --learner
