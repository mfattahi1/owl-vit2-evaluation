import os
import cv2
import json
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.ops import nms

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "./JPEGImages/"            # Path to VOC test images
OUTPUT_JSON = "predictions_owlvit.json" # Output file
CONF_THRESHOLD = 0.3                    # Confidence threshold
IOU_THRESHOLD = 0.5                     # NMS threshold

# -----------------------------
# VOC Class → Prompt Variants
# -----------------------------
ENSEMBLE_PROMPTS = {
    "aeroplane": ["a plane", "an aeroplane", "an airplane", "a jet", "a passenger plane"],
    "bicycle": ["a bicycle", "bike"],
    "bird": ["a bird", "bird"],
    "boat": ["a boat", "ship", "vessel"],
    "bottle": ["a bottle", "bottle", "plastic bottle"],
    "bus": ["a bus", "bus", "public bus"],
    "car": ["a car", "car", "vehicle", "automobile"],
    "cat": ["a cat", "cat", "feline"],
    "chair": ["a chair", "an office chair", "a dining chair", "a wooden chair"],
    "cow": ["a cow", "cow", "cattle"],
    "diningtable": ["a dining table", "a dinner table", "a table with plates"],
    "dog": ["a dog", "dog", "puppy"],
    "horse": ["a horse", "horse", "pony"],
    "motorbike": ["a motorcycle", "motorbike", "bike"],
    "person": ["a person", "a man", "a woman", "a child", "a human"],
    "pottedplant": ["a potted plant", "a houseplant", "an indoor plant", "a green plant in a pot"],
    "sheep": ["a sheep", "a lamb", "a farm animal", "a woolly sheep"],
    "sofa": ["a sofa", "couch", "sofa"],
    "train": ["a train", "train", "locomotive"],
    "tvmonitor": ["a television", "TV", "screen", "monitor"]
}

# -----------------------------
# Load model and processor
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# -----------------------------
# Run model and collect predictions
# -----------------------------
all_predictions = {}
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
#MAX_IMAGES = 2000  # or any number you want
#image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])[:MAX_IMAGES]



for image_file in tqdm(image_files, desc="Running OWL-ViT2 with Prompt Ensembles"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    image = cv2.imread(image_path)[:, :, ::-1]  # Convert BGR to RGB
    image_results = []

    for class_name, prompt_variants in ENSEMBLE_PROMPTS.items():
        combined_boxes = []
        combined_scores = []

        for prompt in prompt_variants:
            inputs = processor(text=[prompt], images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_size = torch.tensor([image.shape[:2]]).to(device)
            results = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_size,
                threshold=CONF_THRESHOLD
            )[0]

            boxes = results["boxes"].detach().cpu()
            scores = results["scores"].detach().cpu()

            if boxes.size(0) > 0:
                combined_boxes.append(boxes)
                combined_scores.append(scores)

        # Apply NMS if we have results
        if combined_boxes:
            boxes = torch.cat(combined_boxes, dim=0)
            scores = torch.cat(combined_scores, dim=0)

            keep_indices = nms(boxes, scores, iou_threshold=IOU_THRESHOLD)
            for idx in keep_indices:
                box = boxes[idx].numpy().tolist()
                score = scores[idx].item()
                image_results.append({
                    "label": class_name,
                    "score": round(score, 3),
                    "box": [round(x, 2) for x in box]
                })

    all_predictions[image_file] = image_results

# -----------------------------
# Save predictions to JSON
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"\n✅ Finished. Saved predictions to: {OUTPUT_JSON}")
