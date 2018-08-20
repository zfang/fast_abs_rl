MODEL=$1
shift
MODE=$1
shift
python decode_full_model.py --path=.output_decode/$MODEL/$MODE --model_dir=pretrained/$MODEL --beam=5 --$MODE $@
