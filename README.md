# Word2vec-on-webtexts
Pytorch implementation for word2vec. <br/>
A pipeline for training word embeddings using word2vec on web-scrapped corpus.

## How to use

* Set training configuration in `configs/config.yaml`
* For training run `python3 -m word2vec.trainer`
* The final embeddings will be saved in `embeddings` folder

## How to use tensorboard projector

* Just run `python3 embedding_projector.py` (it will automatically generate `logs` folder)
* In terminal type `tensorboard --logdir logs/`

## Supported features

* Skip-gram
* Batch update 
* Negative Sampling
* Sub-sampling of frequent word
* Nearest Neigbors search and tensorboard visalization


## To search nearest neigbors

* Just run  `python3 nearest_neighbors.py -word car -topk 5`

Here are some awesome examples

`python3 nearest_neighbors.py -word car -topk 5`
* Top 1 nearest: cars, score 0.67
* Top 2 nearest: automobiles, score 0.58
* Top 3 nearest: vehicle, score 0.58
* Top 4 nearest: mileage, score 0.58
* Top 5 nearest: vehicles, score 0.52

`python3 nearest_neighbors.py -word covid`
* Top 1 nearest: coronavirus, score 0.62
* Top 2 nearest: outbreak, score 0.61
* Top 3 nearest: pandemic, score 0.56
* Top 4 nearest: cmeminsave, score 0.54
* Top 5 nearest: crisis, score 0.53





