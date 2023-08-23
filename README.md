# SMAT - Separable Self and Mixed Attention Transformers for Efficient Object Tracking
The official implementation of **SMAT** [WACV2024]
![SMAT_block](assets/SMAT_block.png)

## News
**`17-08-2023`**: The SMAT tracker training and inference code is released
**`14-08-2023`**: The SMAT is accepted to WACV2024

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
* Place the pretrained backbone model under the `pretrained_models/` folder
* For data preparation, please refer to [this](https://github.com/botaoye/OSTrack/tree/main)
* Uncomment lines `63, 67, and 71` in the [base_backbone.py](https://github.com/goutamyg/SMAT/blob/main/lib/models/mobilevit_track/base_backbone.py) file. 
Long story short: The code is opitmized for high inference speed, hence some intermediate feature-maps are pre-computed during testing. However, these pre-computations are not feasible during training. 
* Run
```
python tracking/train.py --script mobilevitv2_track --config mobilevitv2_256_128x1_ep300 --save_dir ./output --mode single
```
* The training logs will be saved under `output/logs/` folder

## Tracker Evaluation

* Update the test dataset paths in `lib/test/evaluation/local.py`
* Place the pretrained tracker model under `output/checkpoints/` folder 
* Run
```
python tracking/test.py --tracker_name mobilevitv2_track --tracker_param mobilevitv2_256_128x1_ep300 --dataset got10k_test or trackingnet or lasot
```
* Change the `DEVICE` variable between `cuda` and `cpu` in the `--tracker_param` file for GPU and CPU-based inference, respectively  
* The raw results will be stored under `output/test/` folder

## Visualization

## Acknowledgements
* We use the Separable Self-Attention Transformer implementation and the pretrained `MobileViTv2` backbone from [ml-cvnets](https://github.com/apple/ml-cvnets). Thank you!
* Our training code is built upon [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking)
* To generate the evaluation metrics for different datasets (except, server-based GOT-10k and TrackingNet), we use the [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit)

## Citation
