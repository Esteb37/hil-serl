# Demonstrations: drive each episode to success with the gamepad (the classifier or B ends it); failures (X, timeout)
# are dropped. Stops at 20 successes. -> demo_data/
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../record_demos.py "$@" \
    --exp_name=spot_pick_place \
    --successes_needed=20
