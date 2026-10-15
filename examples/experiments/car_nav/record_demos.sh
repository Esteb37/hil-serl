# Demonstrations: press Start, then drive to the red box and park in it (aligned, stopped); X aborts. Failures (off road,
# X, timeout) are dropped. Stops at 20 successes. -> demo_data/
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../record_demos.py "$@" \
    --exp_name=car_nav \
    --successes_needed=20
