# The Overlooked Elephant of Object Detection: Open Set

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
