import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from word2vec.utils import load_cfg
from word2vec.dataset import StanfordSentiment, Word2vecDataset
from word2vec.skip_gram import SkipGram

class Word2VecTrainer:

    def __init__(self, config_path="configs/config.yaml"):

        args = load_cfg(config_path)
        self.args = args
        self.data = StanfordSentiment(args)
        dataset = Word2vecDataset(self.data)

        self.id2token = dataset.sst_data.id2token
        self.dataloader = DataLoader(dataset, batch_size=args["batch_size"],
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.vocab_size = len(self.data.token2id)
        self.emb_dim = args["emb_dim"]

        self.batch_size = args["batch_size"]
        self.iterations = args["epochs"]
        self.initial_lr = args["initial_lr"]

        self.skip_gram_model = SkipGram(self.vocab_size, self.emb_dim)
        self.use_cuda = torch.cuda.is_available() and args["use_cuda"]
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()


    def train(self):

        optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
        path_to_save = os.path.join(self.args["output_folder"],"word_vectors.npy")

        for iteration in tqdm(range(self.iterations)):

            running_loss = 0.0
            for i, sample_batched in enumerate(self.dataloader):
                
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()

                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    
                    scheduler.step()


                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                
            if iteration%20==0:
                print(" Loss: %f" %running_loss)

            self.skip_gram_model.save_embedding(path_to_save)

        json.dump(self.id2token, open(os.path.join(self.args["output_folder"],"index2word.json"),'w'))

tr = Word2VecTrainer()
tr.train()








