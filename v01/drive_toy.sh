cd /workspace/v01
S=toy_status.txt
echo "START $(date -u +%FT%TZ)" > $S
python3 -m pytest tests/ -q > pytest.log 2>&1
echo "pytest_rc=$? :: $(tail -1 pytest.log)" >> $S
echo "CONTROL_START $(date -u +%FT%TZ)" >> $S
python3 -u train.py --mode linear --task mqar --preset toy --steps 3000 --batch 64 --lr 3e-3 --eval_every 250 --eval_trials 100 --tag toy_control_linear > train_control.log 2>&1
echo "control_rc=$?" >> $S
echo "TREAT_START $(date -u +%FT%TZ)" >> $S
python3 -u train.py --mode mlp --task mqar --preset toy --steps 3000 --batch 64 --lr 3e-3 --eval_every 250 --eval_trials 100 --tag toy_treat_mlp > train_treat.log 2>&1
echo "treat_rc=$?" >> $S
echo "ALL_DONE $(date -u +%FT%TZ)" >> $S
