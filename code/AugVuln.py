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
from sklearn.metrics import recall_score, precision_score, f1_score
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

from data_process.textdata import read_answers,read_predictions,calculate_scores,TextDataset
logger = logging.getLogger(__name__)

def test(args, model, tokenizer,test_for_each = False, idx = 0 ):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)        
        label = batch[1].to(args.device)
        seq_inputs = batch[2].to(args.device)
        with torch.no_grad():
            logit = model(seq_inputs, inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    if test_for_each:
        result_file = os.path.join(args.output_dir, "test_for_each_epoch")
    else:
        result_file = os.path.join(args.output_dir, "cvae")

    if not os.path.exists(result_file):
        os.makedirs(result_file)
    with open(os.path.join(result_file, f"predictions-{idx}.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx + '\t1\n')
            else:
                f.write(example.idx + '\t0\n')
    predictions_filename = os.path.join(result_file, f"predictions-{idx}.txt")
    answers_filename = args.test_data_file
    result_filename =  os.path.join(result_file, f"result-{idx}.txt")
    answers = read_answers(answers_filename)
    predictions = read_predictions(predictions_filename)

    # 计算评分
    scores, Result, Fcount, Tcount, TTcount, TTTcount, FFcount, count, Acc = calculate_scores(answers, predictions)      

    with open(result_filename, 'w') as f:
        f.write(f"epoch: {idx}\n")
        f.write("Scores:\n")
        for key, value in scores.items():
            f.write(f"{key}: {value}\n")
        f.write("\nResult:\n")
        for item in Result:
            f.write(f"{item}\n")
        f.write(f"\nFcount: {Fcount}\n")
        f.write(f"Tcount: {Tcount}\n")
        f.write(f"TTcount: {TTcount}\n")
        f.write(f"TTTcount: {TTTcount}\n")
        f.write(f"FFcount: {FFcount}\n")
        f.write(f"count: {count}\n")
        f.write(f"Balanced Accuracy: {Acc}\n") 

def train(args, dataset, model, tokenizer,sample_weight = None,balance = False,):
    """ Train the model """
    train_dataset, eval_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    # 根据 args.local_rank 的值选择适当的数据采样器。RandomSampler 用于随机采样，而 DistributedSampler 用于分布式训练
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # 创建一个数据加载器，它用于加载训练数据批次。这里指定了批次大小、多线程加载数据的数量以及是否将数据加载到固定内存中
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    #  max_steps、save_steps、warmup_steps等
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.cl_epoch
    model.to(args.device)  # 将模型移到指定的设备上，通常是 GPU

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    model.zero_grad()
    # 初始化类别计数
    class_counts = [0, 0]  # 二分类情况下，初始为 [0, 0]
    
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            # 将数据移动到指定的设备
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            seq_inputs = batch[2].to(args.device)
             # 获取当前批次样本的全局索引
            batch_start = step * args.train_batch_size
            batch_end = min((step + 1) * args.train_batch_size, len(train_sampler))
            batch_indices = list(train_sampler)[batch_start:batch_end]
            
            # 统计类别样本数量
            class_counts[0] += (labels == 0).sum().item()
            class_counts[1] += (labels == 1).sum().item()

            # 模型训练
            model.train()

            loss, logits = model(seq_inputs, inputs, labels)
            # 计算损失和预测
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))  # 记录损失和训练步骤等信息

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                
                
                
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results ,logits = evaluateCL(args, model, tokenizer, eval_dataset, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                        # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        # if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
        if (idx + 1) % 100 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{idx}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Evaluate the model on the test dataset after saving
        test(args, model, tokenizer,test_for_each = True ,idx=idx)


def evaluateCL(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    # eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    # 将数据移动到指定的设备
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    # 设置模型为评估模式
    model.eval()
    logits = []
    labels = []

    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        seq_inputs = batch[2].to(args.device)
        model.eval()
        with torch.no_grad():
            lm_loss, logit = model(seq_inputs, inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5


    eval_acc = np.mean(labels == preds)
    eval_f1 = f1_score(labels, preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_f1": round(eval_f1, 4)
    }
    
    print(f"logits shape after flatten: {logits.shape}")
    return result, logits


def get_noise(train_dataset, logits_new, model,args):
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    device = args.device
    for batch in train_dataloader:
        inputs_new = batch[0].to(device)
        labels_new = batch[1].to(device)
        seq_inputs_new = batch[2].to(device)

    labels_new = labels_new.cpu().numpy()
    seq_inputs_new = seq_inputs_new.cpu().numpy()
    inputs_new = inputs_new.cpu().numpy()

    # print("logits_new shape:", logits_new.shape)
    # print("labels_new shape:", labels_new.shape)
    # print("seq_inputs_new shape:", seq_inputs_new.shape)
    # print("inputs_new shape:", inputs_new.shape)

    confident_joint = compute_confident_joint(
        s=labels_new,  # 真实标签
        psx=logits_new,  # P(s = k|x) # 标签预测的概率
        thresholds=None  # 阈值
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=labels_new.astype(np.int16),
        py_method='cnt',
        converge_latent_estimates=False  # 表示不进行潜在标签估计的迭代收敛
    )

    ordered_label_errors = get_noise_indices(
        s=labels_new,
        psx=logits_new,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_class'  # 使用基于类别的剪枝方法
    )

    x_mask = ~ordered_label_errors
    

    # 使用掩码过滤样本
    inputs_pruned = inputs_new[x_mask]
    labels_pruned = labels_new[x_mask]
    seq_inputs_pruned = seq_inputs_new[x_mask]
    
    # 计算样本权重，用于后续的分类模型训练。样本权重根据噪声矩阵中的噪声比例来设置
    sample_weight = np.ones(np.shape(labels_pruned))  # 初始化一个权重数组sample_weight，其形状与剔除噪声后的标签s_pruned相同，初始值全部为1
    for k in range(2):  # 二分类
        sample_weight_k = 1.0 / noise_matrix[k][k]  # 权重的计算方式是取其倒数，这是因为我们希望对于较为容易被错误标记的类别分配较低的权重，从而降低其对模型的影响
        sample_weight[
            labels_pruned == k] = sample_weight_k  # 将对应类别k的样本在sample_weight中的权重更新为sample_weight_k，噪声样本所在的类别将被赋予更低的权重，减小其在模型训练中的影响

    train_dataset.get_new(inputs_pruned,labels_pruned,seq_inputs_pruned)
    # save_process_file(x_mask,args)
    return train_dataset, sample_weight

def trainCL(args, dataset, model, tokenizer):
    """ Train the model """

    # 获取样本数量
    num_samples = len(dataset)

    # 获取索引数组
    indices = list(range(num_samples))

    # 加入十折的交叉验证
    skf = StratifiedKFold(n_splits=args.splits_num, shuffle=True, random_state=args.seed)
    
    # 初始化统计数组
    label_counts_initial = {'0': [], '1': []}
    label_counts_pruned = {'0': [], '1': []}
    
    # 初始化 all_logits 为一个空的二维数组
    all_logits_new = np.empty((0, 2), dtype=np.float32)  # 0行2列的数组，dtype 根据需要调整
    for fold, (train_index, test_index) in enumerate(skf.split(indices, [dataset[i][1].item() for i in indices])):
        # 根据索引划分训练集和验证集
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        eval_dataset = torch.utils.data.Subset(dataset, test_index)
        
        # 统计这一折训练集去噪前的标签分布
        labels_initial = np.array([dataset[i][1].item() for i in train_index])
        label_counts_initial['0'].append(np.sum(labels_initial == 0))
        label_counts_initial['1'].append(np.sum(labels_initial == 1))

        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

        args.max_steps = args.epoch * len(train_dataloader)
        args.save_steps = len(train_dataloader)
        args.warmup_steps = len(train_dataloader)
        args.logging_steps = len(train_dataloader)
        args.num_train_epochs = args.epoch
        model.to(args.device)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                    num_training_steps=args.max_steps)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)

        checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
        scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
        optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
        if os.path.exists(scheduler_last):
            scheduler.load_state_dict(torch.load(scheduler_last))
        if os.path.exists(optimizer_last):
            optimizer.load_state_dict(torch.load(optimizer_last))

        # Train!
        logger.info("***** Running training to get psx  *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.max_steps)

        global_step = args.start_step
        tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
        best_mrr = 0.0
        best_acc = 0.0
        best_f1 = 0.0
        # 初始化类别计数
        class_counts = [0, 0]  # 二分类情况下，初始为 [0, 0]
        model.zero_grad()
        for idx in range(args.start_epoch, int(args.num_train_epochs)):
            all_logits = np.empty((0, 2), dtype=np.float32)
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_num = 0
            train_loss = 0
            for step, batch in enumerate(bar):
                inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                seq_inputs = batch[2].to(args.device)

                
                # 统计类别样本数量
                class_counts[0] += (labels == 0).sum().item()
                class_counts[1] += (labels == 1).sum().item()
                model.train()
                loss, logits =  model(seq_inputs, inputs, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()
                if avg_loss == 0:
                    avg_loss = tr_loss
                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description("epoch {} loss {}".format(idx, avg_loss))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    output_flag = True
                    avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logging_loss = tr_loss
                        tr_nb = global_step

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results, logits = evaluateCL(args, model, tokenizer, eval_dataset, eval_when_training=True)
                            all_logits = np.concatenate((all_logits, logits), axis=0)
                            # if results['eval_acc'] > best_acc:
                            if results['eval_f1'] > best_f1:
                                # if results['eval_acc'] > best_acc:
                                best_acc = results['eval_acc']
                                best_f1 = results['eval_f1']
                                logger.info("  " + "*" * 20)
                                logger.info("  Best acc:%s", round(best_acc, 4))
                                logger.info("  " + "*" * 20)

                                checkpoint_prefix = 'checkpoint-best-acc'
                                output_dir = os.path.join(args.output_dir_before, '{}'.format(checkpoint_prefix))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                                torch.save(model_to_save.state_dict(), output_dir)
                                logger.info("Saving model checkpoint to %s", output_dir)
                            
        all_logits_new = np.concatenate((all_logits_new, all_logits), axis=0)      

    train_dataset, sample_weight = get_noise(dataset, all_logits_new, model, args)
    label_counts_pruned['0'].append(np.sum(train_dataset[1] == 0))
    label_counts_pruned['1'].append(np.sum(train_dataset[1] == 1))


    return train_dataset,sample_weight


