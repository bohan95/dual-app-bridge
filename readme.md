# Dual-app Bridge
#### [Deterministic Image-to-Image Translation via Denoising Brownian Bridge Models with Dual Approximators]()

**Bohan Xiao*, Peiyong Wang*, Qisheng He, Ming Dong**

## Note
The code and documentation here are primarily based on https://github.com/xuekt98/BBDM. We sincerely thank the authors for their contribution.

## Requirements
```commandline
conda env create -f environment.yml
conda activate dual-bridge
```
## Updates
The conditional setup (conditional on Y) is highly recommended, as in our experience, it leads to faster model convergence.

## Data preparation
### Paired translation task
For datasets that have paired image data, the path should be formatted as:
```yaml
your_dataset_path/train/A  # training reference
your_dataset_path/train/B  # training ground truth
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


### Visualizing
You can refer to visualize.ipynb for visualizing the model's sampling process. We will release additional checkpoints in future updates. However, please note that we did not conduct an in-depth exploration of the model’s hyperparameters, as this was not our primary focus.
<!-- ```

