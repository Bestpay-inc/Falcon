import argparse
import os
import random
import secrets  # 添加secrets模块导入
parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='vicuna-7b-v1.3')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=2)
parser.add_argument('--p_w', type=float, default=0.1)
parser.add_argument('--v_w', type=float, default=1.0)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--rep_reduce_gamma', type=float, default=0.5)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--resume_epoch', type=int, default=0)
args = parser.parse_args()

import torch.nn.functional as F
from safetensors.torch import load_file
import logging
import time
def create_logger(args):
    """
    :param logger_file_name:
    :return:
    """
    path=args.cpdir
    path2=path+"/"+args.cpdir.split('/')[-1]
    if not os.path.exists(path):
        os.mkdir(path)

    loca=time.strftime('%Y-%m-%d-%H-%M-%S')
    logger_file_name=path+"/"+args.cpdir.split('/')[-1]+ "_train_log" + ".txt"
    logger = logging.getLogger()         # 设定日志对象
    logger.setLevel(logging.INFO)        # 设定日志等级

    file_handler = logging.FileHandler(logger_file_name)   # 文件输出
    console_handler = logging.StreamHandler()              # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)       # 设置文件输出格式
    console_handler.setFormatter(formatter)    # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = create_logger(args)
logger.info("Start training!")
train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": args.p_w,
    "v_w": args.v_w,
    "head_w": 0.1,
    "num_workers": 1,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    #"noise": "gaussian",
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}

logger.info("========Training Config=========")
logger.info(train_config)
import json
from safetensors import safe_open
from transformers import AutoTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
from models.cnets import Model
from models.configs import FConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig


if accelerator.is_main_process:
    import wandb

    wandb.init(project="test",  config=train_config)

# baseconfig = AutoConfig.from_pretrained(args.basepath)

baseconfig = AutoConfig.from_pretrained(args.basepath, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(args.basepath)

#print('tokenizer.model_max_length:', tokenizer.model_max_length)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

config = FConfig.from_pretrained(train_config["config_path"])

embed_tokens = torch.nn.Embedding(baseconfig.vocab_size, baseconfig.hidden_size, config.pad_token_id)
# print('head:', head)


try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
        
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

try:
    with open(os.path.join(args.basepath,"model.safetensors.index.json"),"r") as f:
        index_json=json.loads(f.read())
        emb_path=index_json["weight_map"]["model.embed_tokens.weight"]
    with safe_open(os.path.join(args.basepath,emb_path),
                               framework="pt",
                               device="cpu") as f:
        tensor_slice = f.get_slice("model.embed_tokens.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            weights=torch.load(os.path.join(args.basepath,emb_path))
            tensor=weights["model.embed_tokens.weight"].float()
embed_tokens.weight.data = tensor
embed_tokens.eval()

for param in head.parameters():
    param.requires_grad = False

for param in embed_tokens.parameters():
    param.requires_grad = False

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data



class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None, k_mask=0):
        self.data = datapath
        self.transform = transform
        self.k_mask = k_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :] 
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]
        
        length = hidden_state.shape[1]
        
        attention_mask = [1] * length
        
        loss_mask = loss_mask[0].tolist()
        
        loss_mask[-self.k_mask:]=[0]*self.k_mask
        
        input_ids_target = input_ids[:, 1:]
        
        input_ids_target2 = input_ids[:, 1+self.k_mask:]
        
        zeropadding1 = torch.tensor([[0]])
        
        input_ids_target = torch.cat((input_ids_target, zeropadding1), dim=1)

        target = hidden_state[:, self.k_mask:, :]
        # print(target.shape)
        zeropadding = torch.zeros(1, self.k_mask, target.shape[2])
        
        target = torch.cat((target, zeropadding), dim=1)
        
        zeropadding = torch.zeros(1, self.k_mask+1)
        
        input_ids_target2 = torch.cat((input_ids_target2, torch.tensor([[0]*(self.k_mask+1)])), dim=1)
        
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        new_data["input_ids_target"] = input_ids_target2
        
        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        
        batch_input_ids_target = torch.cat([self.paddingtensor2D(item['input_ids_target'], max_length) for item in features])
        
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        
        #print('batch_target:', batch_target)
                
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "input_ids_target":batch_input_ids_target,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        
        #print('batch:', batch)
        
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
         
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
            
        return res

@torch.no_grad()
def getkacc(model, data, head, k_mask=2, max_length=5):
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    #print('hidden_states', hidden_states.shape)
    #print('input_ids', input_ids.shape)
    # attention_mask=data["attention_mask"]
    loss_mask = data["loss_mask"]
    # sample_mask=data["sample_mask"]
    target = data["target"]
    total = [1e-5 for _ in range(max_length)]
    correct = [1e-5 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

    for i in range(bs):
        for j in range(k_mask+1, sl):

            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break
                tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1]-k_mask:single_hidden_states.shape[1]]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1]-k_mask:single_hidden_states.shape[1]]
                #print('tmp_in_target_headout', tmp_in_target_headout.shape)
                target_in_token = torch.argmax(tmp_in_target_headout, dim=1)
                #print('target_in_token', target_in_token.shape)
                #raise RuntimeError
                target_out_token = torch.argmax(tmp_out_target_headout, dim=1)
                tmp_token = input_ids[i, single_hidden_states.shape[1]-k_mask:single_hidden_states.shape[1]]
                # tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                #print('(target_in_token == tmp_token).sum()', (target_in_token == tmp_token).sum())
                #print(target_in_token)
                #print(tmp_token)
                if (target_in_token == tmp_token).sum() < k_mask:
                    break
                out_hidden = model(single_hidden_states, input_ids=single_input_ids)
                last_hidden = out_hidden[:, -k_mask:]
                last_headout = head(last_hidden)
                #print('last headout', last_headout.shape)
                token = torch.argmax(last_headout.squeeze(0), dim=1)
                #print('token', token.shape)
                total[k] += k_mask
                if (token == target_out_token).sum()>0:
                    correct[k] += (token == target_out_token).sum().cpu()
                if (token == target_out_token).sum()<k_mask:
                    for kk in range(k + 1, max_length):
                        total[kk] += k_mask
                    break
                #print('token', token[None, None].shape)
                #print('single_input_ids', single_input_ids.shape)
                single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -k_mask:]), dim=1)
                single_input_ids = torch.cat((single_input_ids, token[None].to(single_input_ids.device)),
                                             dim=1)
    #print(total)
    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc
    

if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        print("use Uniform Noise")
        aug = AddUniformNoise(std=train_config["std"])
    else:
        print("use Gaussian Noise")
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]




#print('config:', config)
k_mask = config.k_mask
traindataset = CustomDataset(traindatapath, transform=aug, k_mask=config.k_mask)
testdataset = CustomDataset(testdatapath, k_mask=config.k_mask)


train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)


model = Model(config, load_emb=True, path=args.basepath)
model = model.to(torch.bfloat16)

logger.info('MODEL {}'.format(model))

# for name, param in model.named_parameters():
#     print('name:', name)
#     print('param.size:', param.size())
    
# print('model:', model)

criterion = nn.SmoothL1Loss(reduction="none")

#print('criterion:', criterion)

optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

if args.checkpoint:
    #model_checkpoint = load_file(args.checkpoint+'/model.safetensors')
    model_checkpoint = load_file(args.checkpoint+'/pytorch_model.bin')
    optimizer_checkpoint = torch.load(args.checkpoint+'/optimizer.bin')
    model.load_state_dict(model_checkpoint)
    optimizer.load_state_dict(optimizer_checkpoint)
    print('*** INFO: Training a model from a checkpoint. ***')

num_epochs = train_config["num_epochs"]

num_warmup_steps = train_config["num_warmup_steps"]

total_steps = train_config["total_steps"]

is_warmup = train_config["is_warmup"]


if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    if args.checkpoint:
        scheduler_checkpoint = torch.load(args.checkpoint+'/scheduler.bin')
        scheduler.load_state_dict(scheduler_checkpoint)
    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )

# accelerator.load_state("checkpoints/state_5")
#print('num_epochs:', num_epochs)

idd=0
for epoch in range(num_epochs + 1-args.resume_epoch):
    epoch_save = epoch + args.resume_epoch
    top_3acc = [0 for _ in range(3)]
    # print('top_3acc:', top_3acc)
    
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    num_replace_min=10000
    T=0.95
  
    for batch_idx, data in enumerate(tqdm(train_loader)):
        #print(data)
        #print("batch_idx:",batch_idx)
        seq_len=data["hidden_states"].size(1)
        with accelerator.accumulate(model):
            
            optimizer.zero_grad()

            with torch.no_grad():
                
                predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                out_head = head(predict)
                out_token = torch.argmax(out_head, dim=-1)
                out_logp = nn.LogSoftmax(dim=2)(out_head)
            # with torch.no_grad():
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head) 
                target_p = target_p.detach()
                target_token=torch.argmax(target_head,dim=-1)
                print('target_token:',target_token)     
                target_token_onehot=F.gumbel_softmax(target_head,hard=True)
                embed_tokens.to(data["input_ids"].device)
                target_emb = embed_tokens(target_token)
                token_mask=data["loss_mask"][:, :, None].squeeze(-1)

                num_mask=torch.sum(token_mask==1).item()
                num_same_elements=((out_token==target_token)*token_mask).sum(dim=1).item()
 
                #随机替换
                lambda_start = 0.6
                lambda_end = 0.25
                lambdda = (lambda_start-lambda_end)*(num_epochs-epoch)/num_epochs+0.1
                
                num_replace=(num_mask-num_same_elements)*lambdda
                
                num_replace_min=min(num_replace,num_replace_min)
                
                replace_index=[]
 
                if num_mask == 0:
                    continue
                
                b = torch.nonzero(token_mask[0]==1).squeeze()
                b = b.cpu().numpy().tolist()
                

                if type(b) != list:
                    continue
                
                if int(num_replace) >= len(b):
                    continue
                
                #if k_mask ==2:
                b = b[::k_mask]
                
                if k_mask ==2:
                    random_number = random.sample(b, int(num_replace))
                else:
                    b = b[:-1]
                    random_number = random.sample(b, int(num_replace)//k_mask)
                    
                    for id in range(len(random_number)):
                        length = secrets.randbelow(k_mask-1)  # 使用secrets.randbelow替代random.randint
                        for _ in range(length):
                            random_number.append(random_number[id]+_+1)

                new_hidden_states = data["hidden_states"].clone()
                new_hidden_states[0,random_number,:] = data["target"][0,random_number,:]
                new_input_ids = data["input_ids"].clone()
                new_input_ids[0,random_number] = data["input_ids_target"][0,random_number]
                
            predict = model(new_hidden_states, input_ids=new_input_ids, attention_mask=data["attention_mask"])
            out_head = head(predict)
            out_token = torch.argmax(out_head, dim=-1)

            out_logp = nn.LogSoftmax(dim=2)(out_head)
            
            
            loss_mask = data["loss_mask"][:, :, None]

            plogp = target_p * out_logp
            
            loss_hard = -torch.sum(torch.sum(loss_mask*(target_token_onehot*out_logp),2))/(loss_mask.sum()+1e-5)

            target_pT=nn.Softmax(dim=2)(target_head/T)
            out_pT=nn.Softmax(dim=2)(out_head/T)
            _log=torch.log(target_pT/(out_pT+1e-8))
            KL=target_p*_log
            loss_soft = torch.sum(torch.sum(loss_mask*KL,2))/(loss_mask.sum()+1e-5)
            ploss = 0.9*loss_soft + 0.1*loss_hard

            vloss = criterion(predict, data["target"])
            
            vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)
        
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss

            accelerator.backward(loss)
            
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        
        with torch.no_grad():

            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            
            ct = loss_mask.sum().item()
            
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            
            target = target.view(-1)[loss_mask.view(-1) == 1]
            
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        logger.info('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    if (epoch + 1) % train_config["save_freq"]:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if batch_idx < 10:
                    acces = getkacc(model, data, head, k_mask=config.k_mask, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"])
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                out_head = head(predict)
                out_logp = nn.LogSoftmax(dim=2)(out_head)
                loss_mask = data["loss_mask"][:, :, None]
                plogp = target_p * out_logp
                ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)
                vloss = criterion(predict, data["target"])
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                # print(topkacc)
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1


        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            logger.info('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            logger.info('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch_save+1}",safe_serialization=False)
