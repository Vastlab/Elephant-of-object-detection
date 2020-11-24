# The Overlooked Elephant of Object Detection: Open Set
This repository contains the code for the evaluation approach proposed in the paper [The Overlooked Elephant of Object Detection: Open Set](https://openaccess.thecvf.com/content_WACV_2020/papers/Dhamija_The_Overlooked_Elephant_of_Object_Detection_Open_Set_WACV_2020_paper.pdf)

Our paper may be cited with the following bibtex
```
@inproceedings{dhamija2020overlooked,
  title={The Overlooked Elephant of Object Detection: Open Set},
  author={Dhamija, Akshay and Gunther, Manuel and Ventura, Jonathan and Boult, Terrance},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1021--1030},
  year={2020}
}
```

Evaluation for the wilderness impact curve is now supported using detectron2

### Dataset is expected under `datasets/` in the following structure 
For Pascal VOC:
```
VOC20{07,12}/
  JPEGImages/
```

For MSCOCO:

```
coco/
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

### Training models for the evaluation
In order to run the evaluation please prepare a model trained with the protocol files in this repo.

You may use the following command to train a FasterRCNN model:

```
python main.py --num-gpus 8 --config-file training_configs/faster_rcnn_R_50_FPN.yaml
```

For convenience models trained with the config files in this repo have been provided at: https://vast.uccs.edu/~adhamija/Papers/Elephant/pretrained_models/

### Running the evaluation script

Please ensure your config is correctly set to load the models trained above. You might want to set the `OUTPUT_DIR` detectron2 config

The following command may be used to run the complete evaluation

```python main.py --num-gpus 2 --config-file training_configs/faster_rcnn_R_50_FPN.yaml --resume --eval-only```
