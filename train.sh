MODEL=$1
shift
CUDA=$1
shift
CUDA_VISIBLE_DEVICES=$CUDA python train_abstractor.py --path=pretrained/$MODEL/abstractor $@ &
CUDA_VISIBLE_DEVICES=$(( $CUDA + 1 )) python train_extractor_ml.py --path=pretrained/$MODEL/extractor $@ &
wait
