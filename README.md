# SMAT - Separable Self and Mixed Attention Transformers for Efficient Object Tracking
The official implementation of **SMAT**
![SMAT_block](assets/SMAT_block.png)

## News


## Installation

Install the dependency packages using the environment file `smat_pyenv.yml`.

Generate the relevant files:
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, modify the datasets paths by editing these files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Training

* Set the path of training datasets in `lib/train/admin/local.py`
* Place the pretrained model under the ***pretrained_models/*** folder
* Run
```
python tracking/train.py --script mobilevitv2_track --config mobilevitv2_256_128x1_ep300 --save_dir ./output --mode single
```
* The training logs will be saved under ***output/logs/*** folder

## Tracker Evaluation

* Update the test dataset paths in `lib/test/evaluation/local.py`
* Place the pretrained tracker model under ***output/checkpoints/*** folder 
* Run
```
python tracking/train.py --script mobilevitv2_track --config mobilevitv2_256_128x1_ep300 --save_dir ./output --mode single
```
* The raw results are stored under ***output/test/*** folder

## Visualization


## Acknowledgements


## Citation
