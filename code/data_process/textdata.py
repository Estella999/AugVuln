import numpy as np
import copy
import csv
import os
from tqdm import tqdm, trange
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import json
from torch.utils.data.distributed import DistributedSampler
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
import logging
import torch
from sklearn.metrics import recall_score, precision_score, f1_score,balanced_accuracy_score,matthews_corrcoef,roc_curve,auc
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
import parserTool as ps
from utils.java_cfg import JAVA_CFG
from utils.python_cfg import PYTHON_CFG
from utils.php_cfg import PHP_CFG
from utils.c_cfg import C_CFG

from parserTool import remove_comments_and_docstrings
from parserTool.parse import Lang
import pickle
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger(__name__)


# 提取路径序列 path_sequence 中包含的代码行，并将它们按照代码行的长度升序排序后返回
# 如果没有指定路径序列，它将提取整个源代码的内容
def extract_pathtoken(source, path_sequence):
    seqtoken_out = []  # 用于存储提取的代码行
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if line != 'exit' and (line in source):
                seq_code += source[line]  # 将当前行在源代码 source 中的内容追加到 seq_code 中
        seqtoken_out.append(seq_code)  # 将整个路径的代码添加到 seqtoken_out 列表中
        if len(seqtoken_out) > 5:  # 检查已提取的代码行数量是否超过 5 行
            break
    if len(path_sequence) == 0:  # 即没有指定路径序列
        seq_code = ''
        for i in source:
            seq_code += source[i]  # 将源代码中的每一行追加到 seq_code 中
        seqtoken_out.append(seq_code)  # 对 seqtoken_out 中的代码行按照它们的长度进行升序排序
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 path_source,
                 idx,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.path_source = path_source
        self.idx = str(idx)
        self.label = label


# 其作用是将输入的 JavaScript 代码样本（js）转换为用于模型训练的特征向量
def convert_examples_to_features(js, tokenizer, path_dict, args):
    clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)
    # 用于去除 JavaScript 代码中的注释和文档字符串。'c' 参数表示要处理的是 C 语言风格的注释
    # source
    code = ' '.join(clean_code.split())  # 将清理后的代码按照空格拆分并重新连接为一个字符串，以便进一步处理
    code_tokens = tokenizer.tokenize(code)[
                  :args.block_size - 2]  # 使用指定的分词器 tokenizer 对清理后的代码进行分词，然后截取其中的一部分，最大长度为 args.block_size-2
    source_tokens = [tokenizer.cls_token] + code_tokens + [
        tokenizer.sep_token]  # 创建源代码的标记列表，包括分词器的 CLS 标记、代码标记以及 SEP 标记，以便后续转化为特征向量
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)  # 将源代码标记列表转化为对应的标识符
    padding_length = args.block_size - len(source_ids)  # 计算要添加的填充长度，以确保特征向量的长度达到 args.block_size
    source_ids += [tokenizer.pad_token_id] * padding_length  # 将填充的标识符（PAD 标记）添加到源代码的标识符列表中，以满足指定的 block_size

    if js['idx'] in path_dict:
        # 如果索引 idx 在 path_dict 中存在，
        # 那么从 path_dict 中获取与该索引相关的路径标记列表 path_tokens1 和 CFG（Control Flow Graph）所有路径的信息
        path_tokens1, cfg_allpath = path_dict[js['idx']]
    else:  # 再次调用 remove_comments_and_docstrings 函数，获取清理后的代码和代码字典
        clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)
        # 以获取控制流图
        if args.language == "java":
            g = JAVA_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
        elif args.language == "python":
            g = PYTHON_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.PYTHON)
        elif args.language == "php":
            g = PHP_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.PHP)
        elif args.language == "c":
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
        s_ast = g.parse_ast_file(code_ast.root_node)
        # 获取所有可能的控制流路径信息
        num_path, cfg_allpath = g.get_allpath()
        # 使用 extract_pathtoken 函数提取路径标记，并将其存储在 path_tokens1 中
        path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)

    all_seq_ids = []
    for seq in path_tokens1:
        # 对路径标记进行分词，然后截取最大长度为 args.block_size - 2 的部分
        seq_tokens = tokenizer.tokenize(seq)[:args.block_size - 2]
        # 创建路径标记的标记列表，包括 CLS 标记、路径标记以及 SEP 标记
        seq_tokens = [tokenizer.cls_token] + seq_tokens + [tokenizer.sep_token]
        # 将路径标记的标记列表转化为对应的标识符
        seq_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
        # 计算填充长度,将填充的标识符（PAD 标记）添加到路径标记的标识符列表中
        padding_length = args.block_size - len(seq_ids)
        seq_ids += [tokenizer.pad_token_id] * padding_length
        all_seq_ids.append(seq_ids)

    # 检查 all_seq_ids 中的路径标识符序列数量是否小于指定的 filter_size
    if len(all_seq_ids) < args.filter_size:
        # 多次添加源代码的标识符序列（source_ids）到 all_seq_ids 中，直到达到 filter_size
        for i in range(args.filter_size - len(all_seq_ids)):
            all_seq_ids.append(source_ids)
    else:  # 截取前 filter_size 个路径标识符序列，丢弃超出 filter_size 部分的序列
        all_seq_ids = all_seq_ids[:args.filter_size]
    
     # Print extracted paths and generated features for debugging
    # print(f"Extracted paths for idx {js['idx']}: {path_tokens1}")
    # print(f"Source tokens: {source_tokens}")
    # print(f"Source ids: {source_ids}")
    # print(f"All sequence ids: {all_seq_ids}")
    
    return InputFeatures(source_tokens, source_ids, all_seq_ids, js['idx'], js['target'])
            
            
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, examples=None):
        if examples is None:
            self.examples = []
        else:
            self.examples = examples
        
        pkl_file = open(args.pkl_file, 'rb')
        path_dict = pickle.load(pkl_file)  # pickle 模块从二进制文件中加载数据，将加载的数据存储在 path_dict 变量中

        if file_path is not None:
            with open(file_path) as f:
                for line in f:
                    js = json.loads(line.strip())  # 将当前行的文本解析为 JSON 对象，存储在 js 变量中
                    self.examples.append(convert_examples_to_features(js, tokenizer, path_dict, args))

            
        self.update_features()
        
        if 'train' in file_path:  # 查 file_path 是否包含字符串 'train'，以确定当前数据集是否为训练集
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

        pkl_file.close()
        self.indices = list(range(len(self.examples)))  # 保存所有样本的索引
           
    def __len__(self):
        return len(self.examples)

    def get_indices(self):
        return self.indices  # 返回索引
    def __getitem__(self, i):
        
        example = self.examples[i]
    
        # 将更新后的特征向量和标签分配给示例对象的属性
        input_ids = torch.tensor(self.feature_df.iloc[i].tolist()).clone().detach()
        label = torch.tensor(self.label_df.iloc[i].tolist()).clone().detach()

        # 根据索引 i 返回对应示例的输入标识符（token IDs）、标签和路径源（path_source）作为 Torch 张量
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(
            self.examples[i].path_source)
    def get_new(self,input_ids,label,path_source):
        self.examples = []
        for i in range(len(label)):
            # 创建 InputFeatures 实例并添加到 examples 列表中
            input_features = InputFeatures(
                input_tokens=None,  # 你需要提供对应的数据
                input_ids=input_ids[i],
                path_source=path_source[i],
                idx=i,  # 或者使用合适的索引
                label=label[i]
            )
            self.examples.append(input_features)

        
        self.update_features()
    def update_features(self):
        feature_columns = [f"input_ids_{i}" for i in range(400)]  # Adjust column range if needed
        feature_df = pd.DataFrame([example.input_ids for example in self.examples], columns=feature_columns)
        label_df = pd.DataFrame([example.label for example in self.examples], columns=["label"])
        
        # Scaling features to [0, 1] range
        scaler = MinMaxScaler()
        feature_df_scaled = scaler.fit_transform(feature_df)
        feature_df_scaled = pd.DataFrame(feature_df_scaled, columns=feature_columns)
        
        self.data_df = pd.concat([feature_df_scaled, label_df], axis=1)
        self.feature_df = feature_df_scaled
        self.label_df = label_df
        
    
def save_process_file(x_mask, args):
    # 打开输入的jsonl文件
    with open(args.train_data_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 初始化一个列表，用于存储符合条件的行
    processed_lines = []

#   遍历每一行，并检查对应的x_mask值
    for i, line in enumerate(lines):
        if x_mask[i]:  # 如果x_mask对应的值为True，则保存该行
            processed_lines.append(line)
    output_file =os.path.join(args.output_dir, "train_process_data_file.jsonl")
    # 将符合条件的行写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for processed_line in processed_lines:
            outfile.write(processed_line)
            
            
            
def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

def calculate_scores(answers,predictions):
    Acc=[]
    y_trues,y_preds=[],[]
    Result=[]
    Fcount = 0
    Tcount = 0
    TTcount = 0
    TTTcount = 0
    FFcount = 0
    count = 0
    # print(answers)
    for key in answers:
        if key not in predictions:
            count = count + 1
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
        Acc.append(answers[key]==predictions[key])
        if answers[key] == 1:
            FFcount = FFcount +1
        if answers[key] == 0:
            TTTcount = TTTcount + 1
        if answers[key] != predictions[key]:
            Fcount = Fcount + 1
            Result.append(key)
        if answers[key] == predictions[key]:
            Tcount = Tcount + 1
            if answers[key] == 0:
                TTcount = TTcount + 1
                
    scores={}

    scores['Acc']=np.mean(Acc)
    scores['Recall']=recall_score(y_trues, y_preds, average="binary")
    scores['Prediction']=precision_score(y_trues, y_preds)
    scores['F1']=f1_score(y_trues, y_preds)
    scores['MCC']=matthews_corrcoef(y_trues,y_preds)
    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_preds, pos_label=1)
    scores['AUC'] = auc(fpr, tpr)
    print(2*scores['Recall']*scores['Prediction']/(scores['Recall']+scores['Prediction']))
    return scores, Result, Fcount, Tcount, TTcount, TTTcount, FFcount, count,Acc

