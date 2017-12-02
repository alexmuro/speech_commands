python train.py \
	--background_volume .3 \
	--window_size_ms 20 \
	--how_many_training_steps 50000,10000 \
	--eval_step_interval 1000 \
	--summaries_dir ./vgg_a \
	--train_dir ./vgg_a \
	--data_dir ../gsk_train \
	--save_step_interval 1000

python train.py \
	--background_volume .3 \
	--window_size_ms 20 \
	--how_many_training_steps 50000,10000 \
	--eval_step_interval 1000 \
	--summaries_dir ./vgg_a \
	--train_dir ./vgg_e \
	--data_dir ../gsk_train \
	--batch_size 500
	--save_step_interval 1000
	
python freeze.py \
--window_size_ms 20 \
--start_checkpoint=./log/conv.ckpt-35000 \
--output_file=./frozen/conv1-35000

python train.py --data_dir ../gsk_train --summaries_dir ./train --train_dir ./train --model_architecture vgg_a


