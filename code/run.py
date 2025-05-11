
import argparse
import gc
import logging
import os
import random
import torch
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse
sys.path.append('../AugVuln/')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from pathlib import Path

from tqdm import tqdm
import multiprocessing



from AugVuln import trainCL,train,test
from data_process.ProcessedData import ProcessedData
from models.CVAE.CVAE_model import *
from CGE import CVAESynthesisData,Pipeline
from data_process.textdata import read_answers,read_predictions,calculate_scores,TextDataset
cpu_cont = multiprocessing.cpu_count()
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):          


# Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training and Evaluation
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        
        if args.VCEorCGE:
            train_dataset_CL,sample_weight = trainCL(args, train_dataset, model, tokenizer)
            # train_dataset_CL = TextDataset(tokenizer, args, args.train_process_data_file)
            train_dataset_CL2 = train_dataset_CL
            
            pl = Pipeline(train_dataset_CL)
            train_dataset_CL = pl.run(args)
            train_dataset= train_dataset_CL.update(train_dataset_CL2)
            
            model.to(args.device)
            train(args, train_dataset, model,tokenizer,sample_weight)
        
        else:
            train_dataset_cvae = train_dataset
            train_dataset_cvae2 = train_dataset_cvae
            pl = Pipeline(train_dataset_cvae)
            train_dataset_cvae = pl.run(args)
            train_dataset = train_dataset_cvae.update(train_dataset_cvae2)
            train_dataset_CL,sample_weight = trainCL(args, train_dataset, model, tokenizer)

            model.to(args.device)
            train(args, train_dataset_CL, model,tokenizer,sample_weight)
                

    results = {}
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)

    return results

def auto_configure_output_paths(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir_before = str(output_dir / "before")
    args.train_process_data_file = str(output_dir / "train_process_data_file.jsonl")
    Path(args.output_dir_before).mkdir(parents=True, exist_ok=True)

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # 基本配置
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--language', type=str, default="c")

    # 文件路径
    parser.add_argument('--train_data_file', type=str, required=True)
    parser.add_argument('--test_data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pkl_file', type=str, required=True)

    # 模型
    parser.add_argument('--model_type', type=str, default="roberta")
    parser.add_argument('--tokenizer_name', type=str, required=True)
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--use_resnet', action='store_true', help="Use ResNet-based model if set.")
    # 超参数
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--train_batch_size', type=int, default=24)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--cl_epoch', type=int, default=8)
    parser.add_argument('--splits_num', type=int, default=2)
    parser.add_argument('--encoder_layer_sizes', type=int, default=0)
    parser.add_argument('--decoder_layer_sizes', type=int, default=0)
    parser.add_argument('--latent_size', type=int, default=5)
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--lr', type=float, default=0.005)

    # 设置
    parser.add_argument('--VCEorCGE', type=bool,default=True)
    parser.add_argument('--d_size', type=int, default=128)
    parser.add_argument('--block_size', type=int, default=400)
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--evaluate_during_training', action='store_true')
    parser.add_argument('--do_lower_case', action='store_true')

    # 优化器
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--warmup_steps', type=int, default=0)

    # 保存与日志
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--save_total_limit', type=int, default=None)
    parser.add_argument('--eval_all_checkpoints', action='store_true')

    # 训练环境
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--server_ip', type=str, default='')
    parser.add_argument('--server_port', type=str, default='')
    parser.add_argument('--cnn_size', type=int, default=128)
    parser.add_argument('--filter_size', type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args = auto_configure_output_paths(args) 
    # 根据参数动态导入 Model
    if args.use_resnet:
        from models.resnet import Model
    else:
        from models.model import Model
        
    main(args)


