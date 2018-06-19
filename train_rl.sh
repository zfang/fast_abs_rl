MODEL=$1
shift
python train_full_rl.py --path=pretrained/$MODEL --abs_dir=pretrained/$MODEL/abstractor --ext_dir=pretrained/$MODEL/extractor $@
