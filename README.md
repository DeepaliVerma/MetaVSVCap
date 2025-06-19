# MetaVSVCap
Extended version of the VSVCap that solve the Long-tailed problem of the Video Cptioning task.
MetaVSVCap/
│
├── data/
│   ├── msvd_preprocess.py         # Caption + feature preprocessor
│   └── vocab.py                   # Vocabulary building (with rare word detection)
│
├── models/
│   ├── graphs.py                  # Visual & semantic graph builder
│   ├── gcn_encoder.py            # GCN for encoding graphs
│   ├── pointer_generator.py      # Decoder with generation + copy mechanism
│   └── metavsvcap.py             # Full model
│
├── meta_learning/
│   └── meta_trainer.py           # Episodic meta-learning engine
│
├── train.py                      # Training entry point
├── eval.py                       # Evaluation script (BLEU, CIDEr, etc.)
└── utils.py                      # Losses, metrics, training utils
