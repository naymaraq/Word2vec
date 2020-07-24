# Word2vec-on-webtexts
Pytorch implementation for word2vec. <br/>
A pipeline for training word embeddings using word2vec on web-scrapped corpus.

## How to use

* Set training configuration in `configs/config.yaml`
* For training run `python3 -m word2vec.trainer`
* The final embeddings will be saved in `embeddings` folder

## How to use tensorboard projector?

* Just run `python3 embedding_projector.py` (it will automatically generate `logs` folder)
* In terminal type `tensorboard --logdir logs/`





