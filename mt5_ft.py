# -*- coding:utf-8 _*-

import os
import sys
import time
import math
import random
import argparse
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score)

import torch
from torch import cuda
from transformers import logging
from transformers import (
    MT5TokenizerFast,
    MT5ForConditionalGeneration)

from polynomial_lr_decay import PolynomialLRDecay

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'

model_name = 'google/mt5-base'
tokenizer = MT5TokenizerFast.from_pretrained(model_name)


def read_insts(mode, lang, form, prompt, max_len=200, ups_num=5000):
    """
    Read instances
    """
    src, tgt = [], []
    literal = tokenizer.encode('Literal')
    figure = tokenizer.encode(form.capitalize())
    if len(prompt) > 0:
        prompt = prompt.format(form.capitalize())

    path = 'data/{}_{}_{}.{}'.format(mode, lang, form, '{}')
    with open(path.format(0), 'r') as f0, \
            open(path.format(1), 'r') as f1:
        f0 = f0.readlines()
        f1 = f1.readlines()
        if mode != 'test':

            # Keep label distribution balanced
            if len(f0) > len(f1):
                f0 = f0[:len(f1)]
            else:
                f1 = f1[:len(f0)]

            # upsample
            if mode == 'train' and len(f0) < ups_num:
                f0 = (f0 * math.ceil(ups_num / len(f0)))[:ups_num]
                f1 = (f1 * math.ceil(ups_num / len(f1)))[:ups_num]

        for seqs, label in zip([f0, f1], [literal, figure]):
            for seq in seqs:
                seq = tokenizer.encode(prompt + seq.strip())
                src.append(seq[:min(len(seq) - 1, max_len)] + seq[-1:])
                tgt.append(label)

    return src, tgt


def collate_fn(insts):
    """
    Pad the instance to the max seq length in batch
    """

    pad_id = tokenizer.pad_token_id
    max_len = max(len(inst) for inst in insts)

    batch_seq = [inst + [pad_id] * (max_len - len(inst))
                 for inst in insts]
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


def paired_collate_fn(insts):
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)

    return src_inst, tgt_inst


class MMFLUDataset(torch.utils.data.Dataset):
    """ Seq2Seq Dataset """

    def __init__(self, src_inst, tgt_inst):
        self.src_inst = src_inst
        self.tgt_inst = tgt_inst

    def __len__(self):
        return len(self.src_inst)

    def __getitem__(self, idx):
        return self.src_inst[idx], self.tgt_inst[idx]


def MMFLUIterator(src, tgt, opt, shuffle=True):
    """
    Data iterator for classifier
    """

    loader = torch.utils.data.DataLoader(
        MMFLUDataset(
            src_inst=src,
            tgt_inst=tgt),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=shuffle)

    return loader


def seq2label(seqs, tokenizer):
    pred = []
    for ids in seqs:
        x = tokenizer.decode(
            ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        pred.append(x.strip('</s> '))
    return pred


def evaluate(model, loader, epoch, tokenizer):
    """
    Evaluation function
    """
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model.generate(
                input_ids=src,
                attention_mask=mask,
                num_beams=5,
                max_length=10)
            pred.extend(seq2label(outs, tokenizer))
            true.extend(seq2label(tgt, tokenizer))
    acc = sum([1 if i == j else 0 for i, j in zip(pred, true)]) / len(pred)
    model.train()

    print('[Info] {:02d}-valid: acc {:.4f}'.format(epoch, acc))

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-seed', default=42, type=int, help='random seed')
    parser.add_argument(
        '-lang', nargs='+', help='language names', required=True)
    parser.add_argument(
        '-form', nargs='+', help='figure of speech', required=True)
    parser.add_argument(
        '-prompt', default='', type=str, help='prompt')
    parser.add_argument(
        '-batch_size', default=32, type=int, help='batch size')
    parser.add_argument(
        '-lr', default=1e-4, type=float, help='ini. learning rate')
    parser.add_argument(
        '-log_step', default=100, type=int, help='log every x step')
    parser.add_argument(
        '-epoch', default=80, type=int, help='force stop at x epoch')
    parser.add_argument(
        '-eval_step', default=1000, type=int, help='eval every x step')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    save_path = 'checkpoints/mt5_{}_{}.chkpt'.format(
        '_'.join(opt.lang), '_'.join(opt.form))

    # read instances from input file
    train_src, train_tgt, valid_src, valid_tgt = [], [], [], []
    for lang in opt.lang:
        for form in opt.form:
            path = 'data/train_{}_{}.0'.format(lang, form)

            if not os.path.exists(path):
                continue

            train_0, train_1 = read_insts(
                'train', lang, form, opt.prompt)
            valid_0, valid_1 = read_insts(
                'valid', lang, form, opt.prompt)
            train_src.extend(train_0)
            train_tgt.extend(train_1)
            valid_src.extend(valid_0)
            valid_tgt.extend(valid_1)
            print('[Info] {} insts of train set in {}-{}'.format(
                len(train_0), lang, form))
            print('[Info] {} insts of valid set in {}-{}'.format(
                len(valid_0), lang, form))

    train_loader = MMFLUIterator(train_src, train_tgt, opt)
    valid_loader = MMFLUIterator(valid_src, valid_tgt, opt)

    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device).train()

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=opt.lr, betas=(0.9, 0.98), eps=1e-09)

    scheduler = PolynomialLRDecay(
        optimizer,
        warmup_steps=1000,
        max_decay_steps=10000,
        end_learning_rate=5e-5,
        power=2)

    loss_list = []
    start = time.time()
    eval_acc, tab = 0, 0
    patience = 6
    for epoch in range(opt.epoch):
        for batch in train_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            optimizer.zero_grad()

            mask = src.ne(tokenizer.pad_token_id).long()
            loss = model(src, mask, labels=tgt)[0]
            loss.backward()
            scheduler.step()
            optimizer.step()
            loss_list.append(loss.item())

            if scheduler.steps % opt.log_step == 0:
                lr = optimizer.param_groups[0]['lr']
                print('[Info] {:02d}-{:05d}: loss {:.4f} | '
                      'lr {:.5f} | sec {:.3f}'.format(
                    epoch, scheduler.steps, np.mean(loss_list),
                    lr, time.time() - start))
                loss_list = []
                start = time.time()

            if ((len(train_loader) >= opt.eval_step
                 and scheduler.steps % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step
                        and scheduler.steps % len(train_loader) == 0
                        and scheduler.steps > 1000)
                    or scheduler.steps == 1000):
                valid_acc = evaluate(
                    model,
                    valid_loader,
                    epoch,
                    tokenizer)

                if eval_acc < valid_acc:
                    eval_acc = valid_acc
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == patience:
                        break
    # evaluation
    print('[Info] Evaluation')
    model.load_state_dict(torch.load(save_path))
    for form in opt.form:
        for lang in opt.lang:
            path = 'data/test_{}_{}.0'.format(lang, form)
            if not os.path.exists(path):
                continue
            test_0, test_1 = read_insts(
                'test', lang, form, opt.prompt)
            test_loader = MMFLUIterator(test_0, test_1, opt)
            print('[Info] {} insts of {}-{}'.format(
                len(test_0), lang, form))
            evaluate(
                model,
                test_loader,
                0,
                tokenizer)

if __name__ == '__main__':
    main()
