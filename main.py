import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import pyplot as plt

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from eval import BasicEvalOperations
import WIC
from detectron2.evaluation import inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

logger = logging.getLogger("detectron2")


colors=['b','g','r','c','m']

def plot_WIC(WIC_values, recalls_to_process, wilderness, line_style='--', label='Faster RCNN'):
    for i,recall_thresh in enumerate(recalls_to_process):
        plt.plot(wilderness,WIC_values[:,i],colors[i]+line_style,label=f"{round(recall_thresh, 1)} {label}")

    plt.ylabel('Wilderness Impact',fontsize=15)
    plt.xlabel('Wilderness Ratio',fontsize=15)
    plt.xticks(torch.arange(0,max(wilderness)+0.5,0.5).tolist(),
               [str(_) for _ in torch.arange(0,max(wilderness)+0.5,0.5).tolist()],
               fontsize=10)
    plt.savefig(f'{label}.pdf', bbox_inches='tight')
    plt.show()


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = BasicEvalOperations(dataset_name, cfg, True, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info(f"Image level evaluation complete for {dataset_name}")
            logger.info(f"Results for {dataset_name}")
            WIC.only_mAP_analysis(results[dataset_name]['predictions']['correct'],
                                  results[dataset_name]['predictions']['scores'],
                                  results[dataset_name]['predictions']['pred_classes'],
                                  results[dataset_name]['category_counts'])
    if comm.is_main_process():
        logger.info(f"Combined results for datasets {', '.join(cfg.DATASETS.TEST)}")
        eval_info={}
        eval_info['category_counts'] = results[list(results.keys())[0]]['category_counts']
        eval_info['predictions']={}
        for dataset_name in results:
            for k in results[dataset_name]['predictions'].keys():
                if k not in eval_info['predictions']:
                    eval_info['predictions'][k]=[]
                eval_info['predictions'][k].extend(results[dataset_name]['predictions'][k])
        WIC.only_mAP_analysis(eval_info['predictions']['correct'],
                              eval_info['predictions']['scores'],
                              eval_info['predictions']['pred_classes'],
                              eval_info['category_counts'])
        Recalls_to_process = (0.1, 0.3, 0.5)
        wilderness = torch.arange(0, 5, 0.1).tolist()
        WIC_values,wilderness_processed = WIC.WIC_analysis(eval_info,Recalls_to_process=Recalls_to_process,wilderness=wilderness)
        plot_WIC(WIC_values, Recalls_to_process, wilderness_processed, line_style='--', label='Faster RCNN')
    return


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )

    for protocol_name in ('custom_voc_2007_train', 'custom_voc_2007_val', 'custom_voc_2012_train', 'custom_voc_2012_val', 'custom_voc_2007_test'):
        register_coco_instances(protocol_name,
                                {},
                                f"protocol/custom_protocols/{protocol_name}.json",
                                f"datasets/VOC{protocol_name.split('_')[-2]}/JPEGImages/")
    register_coco_instances("Mixed_Unknowns", {}, "protocol/custom_protocols/Mixed_Unknowns.json", "datasets/coco/train2017")
    register_coco_instances("WR1_Mixed_Unknowns", {}, "protocol/custom_protocols/WR1_Mixed_Unknowns.json", "datasets/coco/train2017")
    register_coco_instances("debug", {}, "protocol/debug.json", "datasets/coco/train2017")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
