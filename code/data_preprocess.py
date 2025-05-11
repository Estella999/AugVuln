import sys
import os
import argparse

import parserTool.parse as ps
from utils.java_cfg import JAVA_CFG
from utils.python_cfg import PYTHON_CFG
from utils.php_cfg import PHP_CFG
from utils.c_cfg import C_CFG

from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import json
import pickle
import logging
import numpy as np
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
tokenizer = tokenizer_class.from_pretrained("./models/codebert", do_lower_case=True)

logger = logging.getLogger(__name__)

def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 10:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out
    
# 处理数据、生成路径
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default=None, help="The language to be processed") # 新加
    parser.add_argument("--data_file", default=None, type=str, required=True,help="The input pre-training data file (a text file).") # 新加
    parser.add_argument('--pkl_file', type=str, default='', help='for dataset path pkl file') # 新加
    args = parser.parse_args()
# short_3path_phpdata_nobalance
    output = open(os.path.join(args.pkl_file, f"short_3path_{args.language}data_nobalance.pkl"), 'wb')
    path_dict = {}
    state_dict = {}
    num_id = 0      # 处理的数据数量。jsonl中，一行-->num_id +1
    sum_ratio = 0   # 路径的占比总和
    num_path_dict = {}   # java_valid
    with open(os.path.join(args.data_file, f"train_{args.language}data.jsonl")) as f:   
        for line in f:
            num_id += 1
            if num_id%100 == 0:
                print(num_id, flush=True)   # flush=True :不经过缓冲区，立即输出到屏幕上
            
            js = json.loads(line.strip())   # 将当前行的json字符串储存到js中

            # code_dict : 原始源代码的行号与内容的映射关系
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)

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
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            sum_ratio += ratio
            path_dict[js['idx']] = path_tokens1, cfg_allpath
    print("train file finish...", flush=True)

    with open(os.path.join(args.data_file, f"valid_{args.language}data.jsonl")) as f:      
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)

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
            num_path, cfg_allpath, _, ratio = g.get_allpath()

            # 提取路径对应的代码tokens
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            sum_ratio += ratio
            path_dict[js['idx']] = path_tokens1, cfg_allpath    # 键：js['idx']   值：path_tokens1, cfg_allpath
    print("valid file finish...", flush=True)

    with open(os.path.join(args.data_file, f"test_{args.language}data.jsonl")) as f:   
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)

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
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            sum_ratio += ratio
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            path_dict[js['idx']] = path_tokens1, cfg_allpath
    print("test file finish...", flush=True)
    print(sum_ratio/num_id, flush=True)
    # Pickle dictionary using protocol 0.
    pickle.dump(path_dict, output)
    output.close()
    
if __name__=="__main__":
    main()