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

### Dataset is expected under `dataset/` in the following structure 
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

The following command may be used to run the complete evaluation

```python main.py --num-gpus 2 --config-file training_configs/faster_rcnn_R_50_FPN.yaml --resume --eval-only```
