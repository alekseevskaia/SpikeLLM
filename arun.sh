#TRAIN
nohup python main.py   --let   --lwc   --wbits 4   --abits 4   --epochs 20   --calib_dataset wikitext2   --nsamples 128 --output_dir ./pretrained/qwen-w4a4-low0.9-20epoch  --batch_size 1 --low_p 0.9  --seed 2   --alpha 0.75 --let_lr 0.001 --lwc_lr 0.01 --tasks "" --eval_ppl   2>&1 | tee generate_20epoch_0.9_2.log

#INFERENCE
python main.py   --let   --lwc   --wbits 4   --abits 4   --epochs 0   --calib_dataset wikitext2   --nsamples 128 --resume ./pretrained/qwen-w4a4-low0.9-20epoch/omni_parameters.pth  --batch_size 1 --low_p 0.9  --seed 2   --alpha 0.75  --tasks "" --chatbot