# OWL-ViT2 Evaluation

## 1. Clone the Repository

```bash
git clone https://github.com/mfattahi1/owl-vit2-evaluation.git
cd owl-vit2-evaluation
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

## 3. Run Inference

Make sure your VOC-formatted dataset is located in a `data/` folder.

```bash
python run_inference.py
```

## 4. Evaluate Predictions

This script computes mAP@0.5 and per-class metrics.

```bash
python evaluate_predictions.py
```

## 📊 Results

| Class        | AP    |
|--------------|-------|
| Cat          | 0.795 |
| Aeroplane    | 0.765 |
| Bird         | 0.747 |
| Dog          | 0.711 |
| Train        | 0.689 |
| Bus          | 0.639 |
| Bottle       | 0.288 |
| Pottedplant  | 0.246 |
| Chair        | 0.274 |
| ...          | ...   |

The full dataset (17,125 images) was evaluated.  
OWL-ViT2 achieved a final **mAP@0.5 of 0.511** without any VOC-specific training.
