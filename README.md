# OWL-ViT2 Evaluation

This repository contains an evaluation of the OWL-ViT2 model for zero-shot object detection on the PASCAL VOC 2012 dataset. Unlike traditional detectors, OWL-ViT2 uses vision-language pretraining and can detect objects from text prompts without retraining.

## 🧠 Overview

- **Zero-shot object detection** using OWL-ViT2
- **Prompt ensembling** for more robust results
- Evaluated on **full PASCAL VOC 2012** (~17k images)
- Reports standard metrics: **Precision, Recall, AP, mAP@0.5**

## 📁 Dataset

We used the Roboflow-hosted version of [PASCAL VOC 2012](https://universe.roboflow.com/jacob-solawetz/pascal-voc-2012/dataset/13), which includes annotations for 20 common object categories (e.g., `person`, `car`, `bottle`, `dog`).

---

## ⚙️ Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/mfattahi1/owl-vit2-evaluation.git
cd owl-vit2-evaluation
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Run Inference
Make sure your VOC-formatted dataset is located in a data/ folder.

bash
Copy
Edit
python run_inference.py
4. Evaluate Predictions
This script computes mAP@0.5 and per-class metrics.

bash
Copy
Edit
python evaluate_predictions.py
📊 Results
Class	AP
Cat	0.795
Aeroplane	0.765
Bird	0.642
Dog	0.711
Bottle	0.288
Chair	0.274
Person	0.448
...	...
mAP@0.5	0.511

🧪 Evaluation Insights
Strong results on visually distinctive classes (e.g., cat, aeroplane)

Weak results on ambiguous or small classes (chair, pottedplant)

Prompt ensembling improved mAP by ~3–4%

📂 File Structure
pgsql
Copy
Edit
owl-vit2-evaluation/
├── run_inference.py
├── evaluate_predictions.py
├── predictions_owlvit.json
├── data/  <-- Put your VOC-style dataset here
├── README.md
└── requirements.txt
📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
OWL-ViT by Google Research

PASCAL VOC 2012

Roboflow for dataset hosting

yaml
Copy
Edit

---


