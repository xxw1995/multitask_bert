import pandas as pd
import numpy as np
import torch
from rdkit import Chem
import re
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 匹配SMILES中的特定字符和组合
smiles_regex_pattern = r'Si|Mg|Ca|Fe|As|Al|Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]icosn]|/|\\'

# SMILES字符串到数字的映射，用于对分子表示进行编码
smiles_str2num = {'<PAD>': 0, 'Cl': 1, 'Br': 2, '#': 3, '(': 4, ')': 5, '+': 6, '-': 7, '0': 8, '1': 9,
    '2': 10, '3': 11, '4': 12, '5': 13, '6': 14, '7': 15, '8': 16, '9': 17, ': ': 18, '=': 19, '@': 20, 'C': 21,
    'B': 22, 'F': 23, 'I': 24, 'H': 25, 'O': 26, 'N': 27, 'P': 28, 'S': 29, '[': 30, ']': 31, 'c': 32, 'i': 33, 'o': 34,
    'Si': 35, 'Mg': 36, 'Ca': 37, 'Fe': 38, 'As': 39, 'Al': 40,
    'n': 41, 'p': 42, 's': 43, '%': 44, '/': 45, '\\': 46, '<MASK>': 47, '<UNK>': 48, '<GLOBAL>': 49, '<p1>': 50,
    '<p2>': 51, '<p3>': 52, '<p4>': 53, '<p5>': 54, '<p6>': 55, '<p7>': 56, '<p8>': 57, '<p9>': 58, '<p10>': 59}

smiles_char_dict = list(smiles_str2num.keys())


def randomize_smile(sml):
    """
    随机化SMILES字符串的函数，使得同一个分子可以有不同的表示
    """
    m = Chem.MolFromSmiles(sml)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans) # # 根据打乱的索引列表重新编号原子
    smiles = Chem.MolToSmiles(nm, canonical=False)
    return smiles


def canonical_smile(sml):
    """
    从分子转换为SMILE格式
    """
    m = Chem.MolFromSmiles(sml)
    smiles = Chem.MolToSmiles(m, canonical=True)
    return smiles


class smiles_bert_dataset(Dataset):
    def __init__(self, path, smiles_head):
        self.df = pd.read_csv(path)
        self.data = self.df[smiles_head].to_numpy().reshape(-1).tolist()
        self.vocab = smiles_str2num

    def __getitem__(self, item):
        smiles = self.data[item]
        char_list = re.findall(smiles_regex_pattern, smiles)
        char_list = ['<GLOBAL>'] + char_list
        nums_list = [self.vocab.get(char_list[j], self.vocab['<UNK>']) for j in range(len(char_list))]

        choices = np.random.permutation(len(nums_list) - 1)[:int(len(nums_list) * 0.15)] + 1  # 15%随机掩码
        y = np.array(nums_list).astype('int64')
        mask = np.zeros(len(nums_list))
        # core
        for i in choices:
            rand = np.random.rand()
            mask[i] = 1  # 被替换的位置设为1，代表将要预测的位置
            if rand < 0.8:
                nums_list[i] = 48  # 替换为[UNK]
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 46 + 0.1)  # 10%替换为其他字符索引
            # 10%保持不变

        x = np.array(nums_list).astype('int64')
        masks = mask.astype('float32')
        # x: 经过可能的掩码和随机更改后的数值编码数组。
        # y: 原始的数值编码数组，这将作为预测时的正确答案（ground truth）
        # masks: 权重数组，指示模型在计算损失时哪些位置是被掩码掉或替换过的，需要模型重点预测
        return x, y, masks

    def __len__(self):
        return len(self.data)


class prediction_dataset(object):
    def __init__(self, df, smiles_head='SMILES', reg_heads=[], clf_heads=[]):
        self.df = df
        self.reg_heads = reg_heads
        self.clf_heads = clf_heads
        self.smiles = self.df[smiles_head].to_numpy().reshape(-1).tolist()
        self.reg = np.array(self.df[reg_heads].fillna(-1000)).astype('float32')
        self.clf = np.array(self.df[clf_heads].fillna(-1000)).astype('int32')

        self.vocab = smiles_str2num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        smiles = self.smiles[item]
        properties = [None, None]
        if len(self.clf_heads)>0:
            clf = self.clf[item]
            properties[0] = clf

        if len(self.reg_heads)>0:
            reg = self.reg[item]
            properties[1] = reg

        nums_list = self._char_to_idx(seq=smiles)
        if len(self.reg_heads) + len(self.clf_heads) >0:
            ps = ['<p{}>'.format(i+1) for i in range(len(self.reg_heads) + len(self.clf_heads))]
            nums_list = [smiles_str2num[p] for p in ps] + nums_list
        x = np.array(nums_list).astype('int32')
        return x, properties

    def numerical_smiles(self, smiles):
        smiles = self._char_to_idx(seq=smiles)
        x = np.array(smiles).astype('int64')
        return x

    def _char_to_idx(self, seq):
        char_list = re.findall(smiles_regex_pattern, seq)
        char_list = ['GLOBAL'] + char_list
        return [self.vocab.get(char_list[j],self.vocab['<UNK>']) for j in range(len(char_list))]

class pretrain_collater():
    """
    准备预训练阶段的数据
    """
    def __init__(self):
        super(pretrain_collater, self).__init__()

    def __call__(self,data):
        xs, ys, masks = zip(*data) # SMILES字符串经过数值化后的数据、预测目标、mask矩阵
        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        ys = pad_sequence([torch.from_numpy(np.array(y)) for y in ys], batch_first=True).long().to(device)
        masks = pad_sequence([torch.from_numpy(np.array(mask)) for mask in masks], batch_first=True).float().to(device)
        return xs, ys, masks


class finetune_collater():
    """
    准备微调阶段的数据
    """
    def __init__(self,args):
        super(finetune_collater, self).__init__()
        self.clf_heads = args.clf_heads
        self.reg_heads = args.reg_heads

    def __call__(self, data):
        xs, properties_list = zip(*data)
        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        properties_dict = {'clf': None, 'reg': None}

        if len(self.clf_heads) > 0:
            """
            如果有分类头，则将每个样本的分类属性组合成一个大的张量。
            """
            properties_dict['clf'] = torch.from_numpy(np.concatenate([p[0].reshape(1,-1) for p in properties_list],0).astype('int32')).to(device)

        if len(self.reg_heads) > 0:
            """
            如果有回归头，则将每个样本的回归属性组合成一个大的张量
            """
            properties_dict['reg'] = torch.from_numpy(np.concatenate([p[1].reshape(1,-1) for p in properties_list],0).astype('float32')).to(device)

        # xs: 填充后的数值化SMILES字符串构成的张量，准备进入模型
        # properties_dict: 包含分类和回归任务数据的字典，每种任务类型对应一个张量。
        return xs, properties_dict
