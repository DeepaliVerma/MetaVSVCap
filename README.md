# MetaVSVCap
Extended version of the VSVCap that solve the Long-tailed problem of the Video Cptioning task.

# MetaVSVCap: Meta-Aware Pointer-Generator Enhanced Video Captioning

MetaVSVCap is an advanced video captioning framework designed to generate informative and diverse natural language descriptions for videos. It enhances the VSVCap architecture by incorporating two key components:

- A **Pointer-Generator Decoder** that enables the use of rare or unseen words by copying tokens from visual and semantic graphs.
- A **Meta-Learning Objective** that simulates head-tail word distributions via episodic training, improving generalization to low-resource and open-domain scenarios.

This implementation supports training and evaluation on the **MSVD (YouTube2Text)** dataset.

---

## 🔧 Features

- Visual-Semantic Graph Construction (object relations + SPO triples)
- GCN-based encoding for multi-modal representation
- Pointer-Generator Decoder for rare term copying
- Meta-Learning (episodic training) for head-tail adaptation
- Evaluation with BLEU, METEOR, ROUGE, CIDEr

---

## 📁 Project Structure

MetaVSVCap/
│
├── data/
│ ├── msvd_preprocess.py # Frame, caption, and graph pre-processing
│ └── vocab.py # Vocabulary construction and frequency tiers
│
├── models/
│ ├── graphs.py # Graph builders for visual & semantic inputs
│ ├── gcn_encoder.py # GCN encoders for graphs
│ ├── pointer_generator.py # Decoder with copy mechanism
│ └── metavsvcap.py # Main model combining all modules
│
├── meta_learning/
│ └── meta_trainer.py # MAML-style episodic training
│
├── train.py # Model training script
├── eval.py # Caption generation + evaluation
├── utils.py # Helper functions for training, decoding
└── README.md # You are here!


---

## 📦 Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- nltk
- spacy
- networkx
- scikit-learn
- tqdm

```bash
pip install -r requirements.txt

----

## Dataset: MSVD (YouTube2Text)
Download from YouTube2Text Dataset

Extract video features using ResNet, C3D, or SlowFast.

Run:
python data/msvd_preprocess.py --video_dir ./MSVD/videos/ --caption_file ./MSVD/descriptions.csv

## Training
python train.py --config configs/msvd_config.json

## Evaluation
python eval.py --model_checkpoint saved/model.pt

##
