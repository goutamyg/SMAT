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

* Update the training dataset paths in `lib/train/admin/local.py`
    
## Tracker Evaluation

* Update the test dataset paths in `lib/test/evaluation/local.py`
    
## Visualization


## Acknowledgements


## Citation
