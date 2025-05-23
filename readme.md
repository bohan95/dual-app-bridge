# Dual-app Bridge
***
#### [Deterministic Image-to-Image Translation via Denoising Brownian Bridge Models with Dual Approximators]()

**Bohan Xiao*, Peiyong Wang*, Qisheng He, Ming Dong**


## Requirements
```commandline
conda env create -f environment.yml
conda activate dual-bridge
```

## Data preparation
### Paired translation task
For datasets that have paired image data, the path should be formatted as:
```yaml
your_dataset_path/train/A  # training reference
your_dataset_path/train/B  # training ground truth
your_dataset_path/val/A  # validating reference
your_dataset_path/val/B  # validating ground truth
your_dataset_path/test/A  # testing reference
your_dataset_path/test/B  # testing ground truth
```
After that, the dataset configuration should be specified in config file as:
```yaml
dataset_name: 'your_dataset_name'
dataset_type: 'custom_aligned'
dataset_config:
  dataset_path: 'your_dataset_path'
```



## Train and Test
### Specify your configuration file
Modify the configuration file based on our templates in <font color=violet><b>configs/Template-*.yaml</b></font>  


### Specity your training and tesing shell
Specity your shell file based on our templates in <font color=violet><b>configs/Template-shell.sh</b></font>


Note that optimizer checkpoint is not needed in test and specifying checkpoint path in commandline has higher priority than specifying in configuration file.

For distributed training, just modify the configuration of **--gpu_ids** with your specified gpus. 
```commandline
python3 main.py --config configs/Template.yaml --sample_to_eval --gpu_ids 0,1,2,3 --resume_model path/to/model_ckpt
```

