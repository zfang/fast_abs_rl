MODEL=$1
MODE=$2
python decode_full_model.py --path=.output_decode/$MODEL/$MODE --model_dir=pretrained/$MODEL --beam=5 --$MODE
