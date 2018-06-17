MODEL=$1
python decode_full_model.py --path=.output_decode/$MODEL/val --model_dir=pretrained/$MODEL --beam=5 --val
python decode_full_model.py --path=.output_decode/$MODEL/val --model_dir=pretrained/$MODEL --beam=5 --test
