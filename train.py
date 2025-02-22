import random
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import os
import datasets
from tqdm import tqdm
import argparse
import wandb
import yaml
import time 

from eval import evaluate

import utils

import numpy as np

from models.loss import global_singular_smoothing_loss, local_singular_smoothing_loss


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data', type=str, default='WebOfScience')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=30)
    # parser.add_argument('--early-stop', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--wandb', default=True, action='store_true')
    parser.add_argument('--arch', type=str, default='bert-base-uncased')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--label-loss-wight', type=float, default=0.0005)
    parser.add_argument('--hie-label-loss-wight', type=float, default=0.0005)
    parser.add_argument('--seloss-wight', type=float, default=0.05)
    # parser.add_argument('--warmup-steps', type=int, default=1000)  # 新增
    # parser.add_argument('--total-steps', type=int, default=10000)  # 新增
    return parser


class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)
        
# def get_scheduler(optimizer, warmup_steps, total_steps): # 线性衰减
#     def lr_lambda(current_step: int):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
#     return LambdaLR(optimizer, lr_lambda)

# def get_cosine_with_warmup_scheduler(optimizer, warmup_steps, total_steps): # 余弦退火
#     def lr_lambda(current_step: int):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         else:
#             # 余弦退火部分
#             progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#             return 0.5 * (1 + torch.cos(progress * 3.141592653589793))
#     return LambdaLR(optimizer, lr_lambda)

def get_exponential_with_warmup_scheduler(optimizer, warmup_steps, total_steps, gamma): # 指数衰减
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            decay_steps = (current_step - warmup_steps) // warmup_steps
            return gamma ** decay_steps
    return LambdaLR(optimizer, lr_lambda)

if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    if args.wandb:
        # wandb.init(config=args, project=str(args.batch) +'_'+ args.data + '_' + 'SIDHTC', name=args.name)
        wandb.init(project=str(args.batch) +'_'+ args.data + '_' + 'SISHTC', name=args.name)
        # args.seloss_wight = wandb.config.seloss_wight
        # args.label_loss_wight = wandb.config.label_loss_wight
        # args.hie_label_loss_wight = wandb.config.hie_label_loss_wight
    print(args)
    utils.seed_torch(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1


    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth


    depth_dict = {i: get_depth(i) for i in range(num_class)} # 每个标签所在的深度
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l)) # 添加虚拟节点的连边

    if args.model == 'prompt':
        if os.path.exists(os.path.join(data_path, args.model)):
            dataset = datasets.load_from_disk(os.path.join(data_path, args.model))
        else:
            dataset = datasets.load_dataset('json',
                                            data_files={'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                        'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                        'test': 'data/{}/{}_test.json'.format(args.data, args.data), })

            prefix = [] # prompt的构建 # tokenizer.vocab_size=30522 num_class=141 max_depth=2
            for i in range(max_depth):
                prefix.append(tokenizer.vocab_size + num_class + i)
                prefix.append(tokenizer.vocab_size + num_class + max_depth)
            prefix.append(tokenizer.sep_token_id) # prefix = [30663, 30665, 30664, 30665, 102]


            def data_map_function(batch, tokenizer): # batch中有token是原句，label是标签的索引
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                for l, t in zip(batch['label'], batch['token']):
                    new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)]) # new_batch['labels']=[[[第一层141],[第二层141]]]
                    for d in range(max_depth): # 每一层
                        for i in depth2label[d]: # 该层的每个标签
                            new_batch['labels'][-1][d][i] = 0
                        for i in l:
                            if new_batch['labels'][-1][d][i] == 0:
                                new_batch['labels'][-1][d][i] = 1
                    new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y] # 282维标签向量 将 new_batch['labels'] 中的最后一个元素从一个二维列表（列表的列表）转换为一维列表（扁平化的列表）

                    tokens = tokenizer(t, truncation=True)
                    new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix) # new_batch['input_ids']添加上模板变为一句话，中间没有102，结尾有102
                    new_batch['input_ids'][-1].extend(
                        [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1]))) # 每个句子添0补充为512维
                    new_batch['attention_mask'].append(
                        tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                    new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                    new_batch['token_type_ids'].append([0] * 512)

                return new_batch


            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(os.path.join(data_path, args.model))
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        from models.prompt import Prompt

    else:
        raise NotImplementedError
    if args.low_res:
        if os.path.exists(os.path.join(data_path, 'low.json')):
            index = json.load(open(os.path.join(data_path, 'low.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(data_path, 'low.json'), 'w'))
        dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])
    model = Prompt.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label, seloss_wight=args.seloss_wight) # from_pretrained是什么意思？怎么用的？
    model.init_embedding()

    model.to('cuda')
    if args.wandb:
        wandb.watch(model) # 假设在命令行中设置了 args.wandb = True，则该代码段将确保在训练过程中，模型的状态和性能指标会被记录和可视化到 wandb 的仪表板中，帮助用户实时了解模型的训练情况和性能表现

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    model.to('cuda')
    optimizer = Adam(model.parameters(), lr=args.lr)
    # scheduler = get_scheduler(optimizer, args.warmup_steps, args.total_steps)  # 新增
    # scheduler = get_cosine_with_warmup_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.total_steps)
    lr_initial = args.lr
    lr_final = 1e-7
    gamma49 = (lr_final / lr_initial) ** (1 / 49)
    if args.data == 'WebOfScience':
        warmup_steps = (30070 // args.batch) + 1
        total_steps = warmup_steps * 50
    elif args.data == 'rcv1':
        warmup_steps = (20833 // args.batch) + 1
        total_steps = warmup_steps * 50
    else: # nyt
        warmup_steps = (23391 // args.batch) + 1
        total_steps = warmup_steps * 50
    scheduler = get_exponential_with_warmup_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, gamma=gamma49)



    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    # early_stop_count = 0
    update_step = 0
    loss = 0
    loss_total = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))
    
    # 计算参数量
    def numParams(net):
        num = 0
        for param in net.parameters():
            if param.requires_grad:
                num += int(np.prod(param.size()))
        return num
    print("numParams:", numParams(model))

    for epoch in range(args.epoch):

        # 用于记录每个 epoch 的 loss 值
        # loss_values = []

        # if early_stop_count >= args.early_stop:
        #     print("Early stop!")
        #     with open('0.0005gl_lrexp_epoch50_agg_32_5_nytseloss.txt', 'a') as f:
        #         f.write(f"Early stop!\n")
        #     # break
        start_time = time.time()
        print("start_time:", start_time)
        model.train()
        with tqdm(train) as p_bar:
            for batch in p_bar: # batch包含'input_ids''attention_mask''labels'
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output, seloss, masked_lm_loss, multiclass_loss = model(**batch) # (**batch)：这是一个 Python 的解包操作符，用于将字典解包为关键字参数。换句话说，如果 batch 字典中有 {'input_ids': tensor1, 'attention_mask': tensor2, ...}，它会相当于 model(input_ids=tensor1, attention_mask=tensor2, ...)。

                # 获取self.weight()的值
                graph_embedding_layer = model.get_input_embeddings()  # 获取GraphEmbedding对象
                label_weight_function = graph_embedding_layer.weight()[-num_class - len(depth2label) - 1: -len(depth2label)-1]

                global_label_loss = global_singular_smoothing_loss(label_weight_function)
                local_label_loss = local_singular_smoothing_loss(label_weight_function, depth2label, num_class)

                loss_total = args.label_loss_wight * global_label_loss + args.hie_label_loss_wight * local_label_loss + output['loss']
                
                # # 记录 loss 值
                # loss_values.append(seloss.item())
                
                # loss_total = output['loss']
                loss_total.backward()
                loss += loss_total.item()
                update_step += 1
                if update_step % args.update == 0:
                    if args.wandb:
                        current_lr = optimizer.param_groups[0]['lr']
                        wandb.log({'epoch': epoch, 'learning_rate': current_lr, 'total_loss': loss, 'se_loss': 0.05*seloss.item(), 'HPT_loss': masked_lm_loss.item() + multiclass_loss.item(), 'masked_lm_loss': masked_lm_loss.item(), 'multiclass_loss': multiclass_loss.item(), 'global_label_loss': args.label_loss_wight * global_label_loss, 'local_label_loss': args.hie_label_loss_wight * local_label_loss})
                    p_bar.set_description(
                        'loss:{:.4f}'.format(loss, ))
                    optimizer.step()
                    scheduler.step()  # 新增
                    optimizer.zero_grad()
                    loss_total = 0
                    loss = 0
                    update_step = 0
                    # torch.cuda.empty_cache()
        end_time = time.time()
        print("end_time:", end_time)
        elapsed_train_time = end_time - start_time
        print(f"epoch train time:{elapsed_train_time}s")
        # # 保存 loss 值到文件中
        # with open('0.0005gl_lrexp_epoch50_agg_32_5_nytseloss.txt', 'a') as f:
        #     # for loss in loss_values:
        #     #     f.write(f"{loss} ")
        #     # f.write(f"\n")
        #     average = sum(loss_values) / len(loss_values)
        #     f.write(f"平均值为: {average}\n")

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev) as pbar:
            start_time = time.time()
            print("start_time:", start_time)
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, ) # output_ids是每个batch预测得到的标签，logits(8,2,141)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
            end_time = time.time()
            print("end_time:", end_time)
            elapsed_eval_time = end_time - start_time
            print(f"epoch eval time:{elapsed_eval_time}s")
            # 获取self.weight()的值
            graph_embedding_layer = model.get_input_embeddings()  # 获取GraphEmbedding对象
            label_weight_function = graph_embedding_layer.weight()[-num_class - len(depth2label) - 1:-1]
            # label_d_loss = -torch.norm(label_weight_function, p='nuc') / torch.sum(
            #     torch.norm(label_weight_function, dim=1))
            # print('label singular loss: ', label_d_loss)
            _, label_s, _ = torch.svd(label_weight_function)
            label_s = label_s.cpu().data.numpy()
            label_s = label_s / np.max(label_s)
            if args.wandb:
                wandb.log({'label_ausc': np.sum(label_s)})
            # print(label_s.tolist())

            # # 输出嵌入矩阵的值
            # print(label_weight_function)

        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)

        # with open('0.0005gl_lrexp_epoch50_agg_32_5_nytseloss.txt', 'a') as f:
        #     f.write(f"macro: {macro_f1}, micro:{micro_f1}\n")

        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1})
        # early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            # early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            # early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
        if args.wandb:
            wandb.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        torch.cuda.empty_cache()

    # test
    test = DataLoader(dataset['test'], batch_size=8, shuffle=False)
    model.eval()


    def test_function(extra):
        checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra)),
                                map_location='cpu')
        model.load_state_dict(checkpoint['param'])
        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)

            # 获取self.weight()的值
            graph_embedding_layer = model.get_input_embeddings()  # 获取GraphEmbedding对象
            label_weight_function = graph_embedding_layer.weight()[-num_class - len(depth2label) - 1:-1]
            label_d_loss = -torch.norm(label_weight_function, p='nuc') / torch.sum(
                torch.norm(label_weight_function, dim=1))
            print('label singular loss: ', label_d_loss)
            _, label_s, _ = torch.svd(label_weight_function)
            label_s = label_s.cpu().data.numpy()
            label_s = label_s / np.max(label_s)
            print('label singular values trend seed3 loss: ', np.sum(label_s))
            print(label_s.tolist())

            # 输出嵌入矩阵的值
            print(label_weight_function)

        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        print("---------------------")
        print(scores['full'])

        # with open('0.0005gl_lrexp_epoch50_agg_32_5_nytseloss.txt', 'a') as f:
        #     f.write(f"TEST macro: {macro_f1}, micro:{micro_f1}\n")

        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)
            prefix = 'test' + extra
        if args.wandb:
            wandb.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})


    test_function('_macro')
    test_function('_micro')

    wandb.finish()

# if __name__ == '__main__':
#     parser = parse()
#     args = parser.parse_args()
#     with open('sweep_config.yaml') as file:
#         sweep_config = yaml.safe_load(file)
#     # 创建 Sweep
#     # sweep_id = wandb.sweep(sweep_config, project=str(args.batch) +'_'+ args.data + '_' + 'SIDHTC')
#     # print("Sweep ID:", sweep_id)
#     # 运行 sweep

#     sweep_id = 'sia4vghk'
#     wandb.agent(sweep_id, project='32_nyt_SIDHTC', function=train)
