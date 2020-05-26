import contextlib
import copy
import io
import itertools
import logging
import os
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.evaluation.evaluator import DatasetEvaluator


class BasicEvalOperations(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2")

        self._metadata = MetadataCatalog.get(dataset_name)
        self.internal_dataset_mapping = copy.deepcopy(self._metadata.thing_dataset_id_to_contiguous_id)
        if -1 in self.internal_dataset_mapping.keys():
            self.internal_dataset_mapping = dict([(k, v - 1) for k, v in self.internal_dataset_mapping.items()])
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
                " Trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

    def reset(self):
        self._predictions = []

    def find_correct_detections(self,detections,ground_truths):
        detected_bbxs = detections['instances'].get('pred_boxes')
        gt_cls_ids = [self.internal_dataset_mapping[gt['category_id']] for gt in ground_truths]
        gt_cls_ids = torch.tensor(gt_cls_ids).to(detected_bbxs.device)

        # To recheck and use the following condition for efficiency
        # if len(detected_bbxs)==0 or len(ground_truths)==0 or set(gt_cls_ids.tolist())==set([-1]):
        if len(detected_bbxs)==0 or len(ground_truths)==0:
            correct = torch.zeros((len(detected_bbxs),), dtype=torch.bool)
            return correct

        pred_classes = detections['instances'].get('pred_classes')

        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in ground_truths
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
        gt_boxes = Boxes(gt_boxes).to(detected_bbxs.device)
        gt_ann_id = [gt['id'] for gt in ground_truths]
        gt_ann_id = torch.tensor(gt_ann_id).to(detected_bbxs.device)

        correct = torch.ones(len(detections['instances']),dtype=torch.bool)
        overlaps = pairwise_iou(detected_bbxs, gt_boxes)
        max_iou, max_iou_indx = torch.max(overlaps, dim=-1)

        correct[max_iou<0.5] = False
        correct[gt_cls_ids[max_iou_indx]!=pred_classes] = False

        # Mark duplicate detections as incorrect
        # navigate through all detections and assign them to a specific annotation/class id
        detected_anns=[]
        correct = correct.tolist()
        for i, (g_ann, correct_status) in enumerate(zip(gt_ann_id[max_iou_indx].tolist(), correct)):
            if g_ann in detected_anns:
                if correct_status: correct[i] = False
            else:
                if correct_status: detected_anns.append(g_ann)

        correct = torch.tensor(correct, dtype=torch.bool)
        return correct

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_contains_mixed_unknowns = [True if _['category_id'] in [-1] else False for _ in self._coco_api.imgToAnns[input['image_id']]]
            if len(image_contains_mixed_unknowns)>0:
                image_contains_mixed_unknowns = torch.any(torch.tensor(image_contains_mixed_unknowns)).item()
            else:
                image_contains_mixed_unknowns = True
            ann_ids = self._coco_api.getAnnIds(imgIds=[input['image_id']])
            annotations = self._coco_api.loadAnns(ann_ids)
            correct = self.find_correct_detections(output,annotations)
            self._predictions.append(dict(image_contains_mixed_unknowns = image_contains_mixed_unknowns,
                                          scores = output['instances'].get('scores'),
                                          correct = correct,
                                          pred_classes = output['instances'].get('pred_classes')))

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        image_contains_mixed_unknowns = [prediction['image_contains_mixed_unknowns'] for prediction in predictions]
        scores = [prediction['scores'] for prediction in predictions]
        correct = [prediction['correct'] for prediction in predictions]
        pred_classes = [prediction['pred_classes'] for prediction in predictions]

        category_counts = {}
        for category in self._coco_api.cats:
            if category not in category_counts:
                category_counts[self.internal_dataset_mapping[category]] = 0
            category_counts[self.internal_dataset_mapping[category]] += len(self._coco_api.getAnnIds(catIds=[category]))

        return dict(predictions = dict(image_contains_mixed_unknowns=image_contains_mixed_unknowns,
                                       scores=scores,
                                       correct=correct,
                                       pred_classes=pred_classes),
                    category_counts = category_counts)

