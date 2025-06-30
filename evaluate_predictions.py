import os
import json
import numpy as np
from xml.etree import ElementTree as ET
from collections import defaultdict
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
PREDICTION_JSON = "predictions_owlvit.json"
ANNOTATIONS_DIR = "./annotations/"  # folder with .xml VOC files
IOU_THRESHOLD = 0.5
VOC_CLASSES = list({
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
})

# -----------------------------
# Load Ground Truth Annotations
# -----------------------------
def load_ground_truths(xml_folder):
    gt_data = defaultdict(list)
    for filename in os.listdir(xml_folder):
        if not filename.endswith(".xml"):
            continue
        image_id = filename.replace(".xml", ".jpg")
        path = os.path.join(xml_folder, filename)
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in VOC_CLASSES:
                continue
            bbox = obj.find("bndbox")
            box = [float(bbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]

            gt_data[image_id].append({"label": name, "box": box})
    return gt_data

# -----------------------------
# IOU Calculation
# -----------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# -----------------------------
# Evaluation Core
# -----------------------------
def evaluate(predictions, ground_truths):
    classwise_results = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in VOC_CLASSES}

    for image_id, preds in tqdm(predictions.items(), desc="Evaluating"):
        gts = ground_truths.get(image_id, [])
        matched = set()

        for pred in preds:
            pred_box = pred["box"]
            pred_label = pred["label"]
            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(gts):
                if gt["label"] != pred_label or idx in matched:
                    continue
                iou = compute_iou(pred_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= IOU_THRESHOLD:
                classwise_results[pred_label]["TP"] += 1
                matched.add(best_gt_idx)
            else:
                classwise_results[pred_label]["FP"] += 1

        # Missed ground truths = FN
        unmatched = [gt for i, gt in enumerate(gts) if i not in matched]
        for gt in unmatched:
            classwise_results[gt["label"]]["FN"] += 1

    # Compute precision, recall, AP
    aps = []
    print("\n--- Results per Class ---")
    for cls, counts in classwise_results.items():
        TP, FP, FN = counts["TP"], counts["FP"], counts["FN"]
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        ap = precision * recall  # Simplified proxy for AP
        aps.append(ap)
        print(f"{cls:12s} | Precision: {precision:.3f} | Recall: {recall:.3f} | AP: {ap:.3f}")

    mean_ap = sum(aps) / len(VOC_CLASSES)
    print(f"\n✅ mAP@0.5 (20 classes): {mean_ap:.3f}")

# -----------------------------
# Run It
# -----------------------------
with open(PREDICTION_JSON, "r") as f:
    predictions = json.load(f)

ground_truths = load_ground_truths(ANNOTATIONS_DIR)
evaluate(predictions, ground_truths)
