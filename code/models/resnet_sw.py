
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.autograd as autograd
from models.Base import BaseModel

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNetTextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        # CNN Part
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

        # Residual Part
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
            for _ in range(2)  # Number of residual blocks, adjust as needed
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def init_params(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight.data)
            nn.init.constant_(conv.bias.data, 0.1)

    def forward(self, x):
        # x: [batch size, sent len, emb dim]
        embedded = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # Apply residual blocks
        residual_out = [conv for conv in conved]
        for res_block in self.residual_blocks:
            residual_out = [res_block(conv) for conv in residual_out]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in residual_out]
        cat = self.dropout(torch.cat(pooled, dim=1))    # [B, n_filters * len(filter_sizes)]
        return self.fc(cat)

class CNNClassificationSeq(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        self.dense = nn.Linear(2*self.d_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.W_w = nn.Parameter(torch.Tensor(2*config.hidden_size, 2*config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(2*config.hidden_size, 1))
        self.linear = nn.Linear(self.args.filter_size*config.hidden_size, self.d_size)
        self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, 3, bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)

        # ResNetTextCNN
        self.window_size = self.args.cnn_size
        self.filter_size = []
        for i in range(self.args.filter_size):
            i = i+1
            self.filter_size.append(i)
            
        self.cnn = ResNetTextCNN(config.hidden_size, self.window_size, self.filter_size, self.d_size, 0.2)
        
        self.linear_mlp = nn.Linear(6*config.hidden_size, self.d_size)
        self.linear_multi = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, features, **kwargs):
        # ------------- CNN with ResNet -------------------------
        x = torch.unsqueeze(features, dim=1) # [B, L*D] -> [B, 1, D*L]
        x = x.reshape(x.shape[0], -1, 768)     # [B, L, D]
        outputs = self.cnn(x)                  # [B, D]
        #features = self.linear_mlp(features)       # [B, L*D] -> [B, D]
        features = self.linear(features)
        x = torch.cat((outputs, features), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- CNN with ResNet ----------------------

        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(BaseModel):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1)        # 3->5

        self.cnnclassifier = CNNClassificationSeq(config, self.args)

    def forward(self, seq_ids=None, input_ids=None, labels=None, sample_weight=None,batch_indices=None):       
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        token_len = seq_ids.shape[-1]

        seq_inputs = seq_ids.reshape(-1, token_len)                                 # [4, 3, 400] -> [4*3, 400]
        seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]    # [4*3, 400] -> [4*3, 400, 768]
        seq_embeds = seq_embeds[:, 0, :]                                           # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)                           # [4*3, 768] -> [4, 3*768]

        logits_path = self.cnnclassifier(outputs_seq)

        prob_path = torch.sigmoid(logits_path)
        prob_path_extended = torch.cat([prob_path, 1 - prob_path], dim=1)
        prob = prob_path_extended
        
        if labels is not None:
            labels = labels.float()
            
            # 原始的损失计算
            loss_original = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob[:, 0]) + 1e-10) * (1 - labels)
            loss_original = -loss_original.mean()

            # 如果 sample_weight 存在并且 batch_indices 存在，则将其应用于损失计算
            if sample_weight is not None and batch_indices is not None:
                # 将 sample_weight 从 numpy array 转换为 PyTorch tensor
                sample_weight = torch.tensor(sample_weight).float()

                # 根据索引提取当前批次的样本权重
                sample_weight_batch = sample_weight[batch_indices]  # 提取权重
                sample_weight_batch = sample_weight_batch.to(labels.device)  # 将权重移动到 GPU（如果在 GPU 上训练）
                
                loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # 使用 'none' 返回逐个样本的损失
                loss_per_sample = loss_fn(prob[:, 0], labels)
                loss = (loss_per_sample * sample_weight_batch).mean()

                return loss, prob
            
            else:
                return loss_original, prob
                
        else:
            return prob