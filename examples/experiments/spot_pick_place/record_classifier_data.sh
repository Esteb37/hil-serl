# Reward classifier data: drive with the gamepad, hold B while the plushy lies in the bin (positives), everything else
# is negative. Stops at 200 positives; Save (GUI) or Ctrl-C writes what is there. -> classifier_data/
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
python ../../record_success_fail.py "$@" \
    --exp_name=spot_pick_place \
    --successes_needed=200
