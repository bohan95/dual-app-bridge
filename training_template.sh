python3 main.py --config configs/Dual_eval.yaml --train --sample_at_start --save_top --gpu_ids 1 
python3 main.py --config configs/Dual_eval.yaml --sample_to_eval --gpu_ids 0 --resume_model results/Cityscapes/ABridge_deep/checkpoint/last_model.pth
python3 main.py --config configs/LinearABridge_temp.yaml --train --sample_at_start --save_top --gpu_ids 1 
python3 main.py --config configs/LinearABridge_temp_noise.yaml --train --sample_at_start --save_top --gpu_ids 1 
python3 main.py --config configs/LinearABridge_temp_noise_without_condition.yaml --train --sample_at_start --save_top --gpu_ids 1 
python3 main.py --config configs/LinearABridge_temp_noise_with_kl.yaml --train --sample_at_start --save_top --gpu_ids 0 

python3 main.py --config configs/LinearABridge_temp.yaml --train --sample_at_start --save_top --gpu_ids 1 
python3 main.py --config configs/LinearABridge_temp_noise.yaml --train --sample_at_start --save_top --gpu_ids 1 

python3 main.py --config configs/ABridge_temp.yaml --train --sample_at_start --save_top --gpu_ids 0 
python3 main.py --config configs/ABridge_temp_128.yaml --train --sample_at_start --save_top --gpu_ids 1 
python3 main.py --config configs/ABridge_temp_noise_128.yaml --train --sample_at_start --save_top --gpu_ids 0 
python3 main.py --config configs/ABridge_temp_noise.yaml --train --sample_at_start --save_top --gpu_ids 0 
python3 main.py --config configs/ABridge_temp_noise.yaml --train --sample_at_start --save_top --gpu_ids 0 
