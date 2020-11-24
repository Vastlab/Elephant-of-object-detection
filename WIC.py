import logging
import torch
import numpy as np
from matplotlib import pyplot as plt
import pathlib

logger = logging.getLogger("detectron2")

# Plot Precision Recall graphs
def PR_plotter(Precision, Recall, cls_name, ap):
    plt.subplots()
    plt.plot(Recall, Precision, 'b', label=f"{round(ap.item() * 100, 2)}%")
    plt.title(cls_name)
    plt.ylabel('Precision', fontsize=20)
    plt.xlabel('Recall', fontsize=20)
    plt.axis([0, 1, 0, 1])
    plt.legend(loc=3, prop={'size': 25}, frameon=False)
    file_name = pathlib.Path(f"PR/{cls_name}_Precision_recall.pdf")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name, bbox_inches="tight")


def calculate_precision_recall(TP_vs_FP, scores, total_no_of_pos):
    scores, sorted_indx = torch.sort(scores, dim=-1, descending=True)
    TP_vs_FP = TP_vs_FP[sorted_indx]
    TP_vs_FP = TP_vs_FP.cumsum(dim=0)
    TP_vs_FP = TP_vs_FP.type(torch.FloatTensor)
    Recall = TP_vs_FP / total_no_of_pos
    # Precision here is non monotonic
    Precision = TP_vs_FP / torch.arange(1, TP_vs_FP.shape[0] + 1, 1).type(torch.FloatTensor)

    Recall = [0.] + Recall.tolist() + [1.]
    Precision = [0.] + Precision.tolist() + [0.]

    # make precision monotonic
    for index_ in range(len(Precision) - 1, 0, -1):
        Precision[index_ - 1] = max(Precision[index_ - 1], Precision[index_])

    Recall = torch.tensor(Recall)
    Precision = torch.tensor(Precision)
    return Precision, Recall

def only_mAP_analysis(correct, scores, pred_classes, category_counts, categories = None):
    correct = torch.cat(correct)
    scores = torch.cat(scores)
    pred_classes = torch.cat(pred_classes)
    all_ap=[]
    for cls_no in set(pred_classes.tolist()):
        Precision, Recall = calculate_precision_recall(correct[pred_classes==cls_no],
                                                       scores[pred_classes==cls_no],
                                                       category_counts[cls_no])
        # 11 point PASCAL VOC evaluation
        ap = []
        for thresh in torch.arange(0, 1.1, 0.1):
            ap.append(torch.max(Precision[Recall >= thresh]))
        ap = torch.mean(torch.tensor(ap))
        if categories is not None:
            PR_plotter(Precision, Recall, categories[cls_no+1]['name'], ap)
        all_ap.append(ap)

        logger.info(f"AP for class no. {int(cls_no)}: {ap}")
    logger.info(f"mAP: {torch.mean(torch.tensor(all_ap))}")

def WIC_analysis(eval_info,Recalls_to_process,wilderness):
    for k in eval_info['predictions']:
        eval_info['predictions'][k] = np.array([torch.tensor(_).type(torch.FloatTensor).numpy() for _ in eval_info['predictions'][k]])

    no_of_closedSetImages = sum(1-eval_info['predictions']['image_contains_mixed_unknowns'])
    mixed_unknowns = eval_info['predictions']['image_contains_mixed_unknowns'].astype(np.bool)
    closed_set_samples = (1 - mixed_unknowns).astype(np.bool)
    eval_predictions = eval_info['predictions']
    WIC_precision_values=[]
    wilderness_processed=[]
    for wilderness_level in wilderness:
        no_of_mixed_unknown_images = int(wilderness_level*no_of_closedSetImages)
        if no_of_mixed_unknown_images>len(eval_predictions['correct'][mixed_unknowns]):
            break
        logger.info(f"{f' Performance at Wilderness level {wilderness_level:.2f} '.center(90, '*')}")
        wilderness_processed.append(wilderness_level)
        correct = eval_predictions['correct'][closed_set_samples].tolist() + \
                  eval_predictions['correct'][mixed_unknowns][:no_of_mixed_unknown_images].tolist()
        scores = eval_predictions['scores'][closed_set_samples].tolist() + \
                 eval_predictions['scores'][mixed_unknowns][:no_of_mixed_unknown_images].tolist()
        pred_classes = eval_predictions['pred_classes'][closed_set_samples].tolist() + \
                       eval_predictions['pred_classes'][mixed_unknowns][:no_of_mixed_unknown_images].tolist()
        correct = torch.cat([torch.tensor(_) for _ in correct])
        scores = torch.cat([torch.tensor(_) for _ in scores])
        pred_classes = torch.cat([torch.tensor(_) for _ in pred_classes])
        all_ap = []
        current_WIC_precision_values = []
        for cls_no in set(pred_classes.tolist()):
            Precision, Recall = calculate_precision_recall(correct[pred_classes==cls_no],
                                                           scores[pred_classes==cls_no],
                                                           eval_info['category_counts'][cls_no])
            class_precisions_at_recall = []
            for recall_thresh in Recalls_to_process:
                class_precisions_at_recall.append(torch.max(Precision[Recall >= recall_thresh]).item())
            current_WIC_precision_values.append(class_precisions_at_recall)
            ap = []
            for thresh in torch.arange(0, 1.1, 0.1):
                ap.append(torch.max(Precision[Recall >= thresh]))
            ap = torch.mean(torch.tensor(ap))
            all_ap.append(ap)
            logger.info(f"AP for class no. {int(cls_no)} at wilderness {wilderness_level:.2f}: {ap}")
        current_WIC_precision_values = torch.tensor(current_WIC_precision_values)
        current_WIC_precision_values = torch.mean(current_WIC_precision_values, dim=0).tolist()
        WIC_precision_values.append(current_WIC_precision_values)
        logger.info(f"mAP at wilderness {wilderness_level:.2f}: {torch.mean(torch.tensor(all_ap))}")
    WIC_precision_values = torch.tensor(WIC_precision_values)
    WIC_precision_values = WIC_precision_values[0,:]/WIC_precision_values
    WIC_precision_values = WIC_precision_values -1
    return WIC_precision_values,wilderness_processed
