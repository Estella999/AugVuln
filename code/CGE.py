
from __future__ import absolute_import, division, print_function

import argparse
import gc
import logging
import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from tqdm import tqdm
import multiprocessing

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

sys.path.append('../AugVuln/')


from models.model import Model
from data_process.ProcessedData import ProcessedData
from models.CVAE.CVAE_model import *

cpu_cont = multiprocessing.cpu_count()
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

class CVAESynthesisData(ProcessedData):
    def __init__(self, raw_data):
        if raw_data is None:
            raise ValueError("raw_data provided to CVAESynthesisData is None")
        super().__init__(raw_data)
        self.rest_columns = raw_data[2]

    def process(self,args):
        print("Starting process method in CVAESynthesisData")
        print(f"Length of label_df: {len(self.label_df)}")
        if len(self.label_df) < 2:
            print("Not enough data to proceed with process.")
            return
        
        equal_zero_index = (self.label_df != 1).values
        equal_one_index = ~equal_zero_index

        pass_feature = np.array(self.feature_df[equal_zero_index])
        fail_feature = np.array(self.feature_df[equal_one_index])

        diff_num = len(pass_feature) - len(fail_feature)

        if abs(diff_num) < 1:
            print("Not enough class difference to proceed with CVAE.")
            return

        generate_for_class_one = diff_num > 1
        num_samples_to_generate = abs(diff_num)
        
        print("Start CVAE!!")
        min_batch = 40
        batch_size = min_batch if len(self.label_df) >= min_batch else len(self.label_df)
        torch_dataset = TensorDataset(torch.tensor(self.feature_df.values, dtype=torch.float32),
                                           torch.tensor(self.label_df.values, dtype=torch.int64))
        loader = DataLoader(dataset=torch_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 )
        input_dimension = len(self.feature_df.values[0])
        hidden_dimension = math.floor(math.sqrt(input_dimension))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.encoder_layer_sizes = [input_dimension, hidden_dimension]
        args.decoder_layer_sizes = [hidden_dimension, input_dimension]

        cvae = CVAE(encoder_layer_sizes=args.encoder_layer_sizes,
                    latent_size=args.latent_size,
                    decoder_layer_sizes=args.decoder_layer_sizes,
                    conditional=args.conditional,
                    num_labels=2).to(device)
        optimizer = torch.optim.Adam(cvae.parameters(), lr=args.lr)
        EPOCH = 1000
        for epoch in range(EPOCH):
            cvae.train()
            train_loss = 0
            for step, (x, y) in enumerate(loader):
                x = x.unsqueeze(0).unsqueeze(0).to(device)
                y = y.unsqueeze(0).unsqueeze(0).to(device)

                recon_x, mu, logvar, z = cvae(x, y)

                loss = loss_fn(recon_x, x, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if epoch % 100 == 0:
                print('====>CVAE training... Epoch: {} Average loss: {:.4f}'.format(epoch,
                                                                                   train_loss / len(loader.dataset)))

        with torch.no_grad():
            c = torch.full((num_samples_to_generate,), 1 if generate_for_class_one else 0, dtype=torch.long).unsqueeze(1).to(device)
            z = torch.randn([c.size(0), args.latent_size]).to(device)
            x = cvae.inference(z, c=c).to("cpu").numpy()
        features_np = np.array(self.feature_df)
        compose_feature = np.vstack((features_np, x))

        label_np = np.array(self.label_df)
        if diff_num >= 0:
            gen_label = np.ones(diff_num).reshape((-1, 1))
        else:
            gen_label = np.zeros(-diff_num).reshape((-1, 1))
        compose_label = np.vstack((label_np.reshape(-1, 1), gen_label))

        self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
        self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)


        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
    
    def update(self, train_dataset):
      train_dataset.feature_df = self.feature_df
      train_dataset.label_df = self.label_df
      return train_dataset


class Pipeline:
    def __init__(self, dataloader):
        if dataloader is None or not hasattr(dataloader, 'data_df'):
            raise ValueError("Provided dataloader is not valid or missing data_df attribute")
        self.dataloader = dataloader

    def run(self,args):
        print("Dataloader is set with data_df attribute.")
        self.data_obj = CVAESynthesisData(self.dataloader)
        self.data_obj.process(args)
        return self.data_obj


