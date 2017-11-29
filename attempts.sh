python train.py /
	--background_volume .3 /
	--window_size_ms 20 /
	--how_many_training_steps 30000,5000 /
	--eval_step_interval 1000 /
	--summaries_dir ./log /
	--train_dir ./log /
	--save_step_interval 1000
	
python freeze.py \
--window_size_ms 20 \
--start_checkpoint=./log/conv.ckpt-35000 \
--output_file=./frozen/conv1-35000

python train.py --summaries_dir ./train --train_dir ./train 


