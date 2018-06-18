MODEL=$1
MODE=$2
python eval_full_model.py --rouge --decode_dir=.output_decode/$MODEL/$MODE
python eval_full_model.py --meteor --decode_dir=.output_decode/$MODEL/$MODE
