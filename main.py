import argparse
from utils.data import Data
import pickle
from utils.batchify_with_label import batchify_with_label
from utils.metric import get_ner_fmeasure
import time
import sys
import torch.optim as optim
from model.bilstm_crf import BiLSTM_CRF
import random
import numpy as np
import torch
import gc
import os

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
torch.cuda.manual_seed(1)

#数据
def data_initialization(data, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()
    return data


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    # 用于衰减学习率
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def evaluate(data, model, name, padding_label):
    ## 评价函数
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu, padding_label)
        tag_seq = model(batch_word, mask, batch_label, batch_wordlen)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results


def train(data, model, save_model_dir, padding_label, seg=True):
    # 训练函数
    #从model中获得所有的可调参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("opimizer is Adam")
    optimizer = optim.Adam(parameters, lr=data.HP_lr, weight_decay=data.weight_decay)
    # print("opimizer is SGD")
    # optimizer = optim.SGD(parameters, lr=data.HP_lr, weight_decay=data.weight_decay)
    best_dev = -1
    best_test = -1
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        batch_loss = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        # 对输入做batch处理
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            model.zero_grad()
            # tensor化
            batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu, padding_label)
            instance_count += 1
            # 对将一个batch经过tensor化后的tensor输入到模型中
            loss, tag_seq = model.neg_log_likelihood(batch_word, mask, batch_label, batch_wordlen)
            # 结果一些辅助信息
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.item()
            total_loss += loss.item()
            batch_loss += loss
            # loss的反传及模型参数的优化
            loss.backward()
            optimizer.step()
            model.zero_grad()
            # 辅助信息
            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        # 在dev集上评价
        speed, acc, p, r, f, _ = evaluate(data, model, "dev", padding_label)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if seg:
            current_score = f
            print(
                "Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))
        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = save_model_dir + '_' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

        # 在test集上评价
        speed, acc, p, r, f, _ = evaluate(data, model, "test", padding_label)
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            current_score_test = f
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        if current_score_test > best_test:
            if seg:
                print("Exceed previous best f score:", best_test)
            else:
                print("Exceed previous best acc score:", best_test)
            # model_name = save_model_dir + '_' + str(idx) + ".model"
            # torch.save(model.state_dict(), model_name)
            # print(model_name)
            best_test = current_score_test
        gc.collect()


if __name__ == '__main__':
    seed_num = 100
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.cuda.manual_seed(seed_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## weibo
    train_file = "data/Weibo/weiboNER.train"
    dev_file = "data/Weibo/weiboNER.dev"
    test_file = "data/Weibo/weiboNER.test"


    word_emb_file = "data/gigaword_chn.all.a2b.uni.ite50.vec"
    print(train_file)
    data = Data()
    data.HP_gpu = False #是否使用GPU
    data.norm_gaz_emb = False #词向量是否归一化
    data.HP_fix_gaz_emb = True #词向量表大小是否固定
    data.HP_bilstm = True
    data.random_seed = seed_num

    # 整体参数设定位置
    data.HP_lr = 0.01
    data.HP_lr_decay = 0.01
    data.HP_iteration = 150
    data.HP_batch_size = 20
    data.gaz_dropout = 0.4
    data.weight_decay = 0.00000005
    data.use_clip = False  #是否控制梯度
    data.HP_clip = 30 #最大梯度
    # LSTM参数
    data.HP_hidden_dim = 300
    data.HP_dropout = 0.7
    data_initialization(data, train_file, dev_file, test_file)
    data.build_word_pretrain_emb(word_emb_file)
    print('finish loading')
    data.generate_instance(train_file, 'train')
    print("train_file done")
    data.generate_instance(dev_file, 'dev')
    print("dev_file done")
    data.generate_instance(test_file, 'test')
    print("test_file done")
    print('random seed: ' + str(seed_num))
    # 模型的声明
    model = BiLSTM_CRF(data)
    print("打印模型可优化的参数名称")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    save_model_dir = "data/model_para/"
    o_label2index = data.label_alphabet.instance2index['O']
    train(data, model, save_model_dir, o_label2index)