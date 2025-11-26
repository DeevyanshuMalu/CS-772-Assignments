unset HF_HUB_CACHE

CFG_SCALES=(1.0 1.4 1.8)
CKPTS=(0 2321 4642 6963)

echo "Evaluating Baseline and SVDD models"
python eval.py --ckpt_num 0 --cfg_scale 1.0 --baseline
python eval.py --ckpt_num 0 --cfg_scale 1.0 --baseline
for CFG_SCALE in "${CFG_SCALES[@]}"; do
  for CKPT in "${CKPTS[@]}"; do
    echo "Evaluating checkpoint $CKPT with CFG scale $CFG_SCALE"
    python eval.py --ckpt_num $CKPT --cfg_scale $CFG_SCALE
    python eval.py --ckpt_num $CKPT --cfg_scale $CFG_SCALE --svdd
  done
done