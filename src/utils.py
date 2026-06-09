import pandas as pd
import re
from collections import defaultdict

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



class RuleLabelMapping():
    def __init__(self, rule_file) -> None:      
        df_rules = pd.read_csv(rule_file, sep='\t')

        label_to_contexts = defaultdict(set)

        columns_to_check = ['Mandatory', 'Accessory', 'Forbidden', 'Neutral']

        for _, row in df_rules.iterrows():
            subsystem = row['Subsystem']
            
            for col in columns_to_check:
                if pd.notna(row[col]):
                    # 提取该列下的所有基因
                    genes = [g.strip() for g in str(row[col]).split(',')]
                    for gene in genes:
                        label_to_contexts[gene].add((subsystem, col))

        # 去掉末尾数字 对所有标签进行分组
        prefix_to_labels = defaultdict(list)
        for label in label_to_contexts.keys():
            # 正则：去掉末尾的下划线和纯数字 (如 Menshen__NsnC_2729199380 -> Menshen__NsnC)
            prefix = re.sub(r'_[0-9]+$', '', label)
            prefix_to_labels[prefix].append(label)


        label_mapping = {}
        protected_prefixes = set()

        for prefix, variants in prefix_to_labels.items():
            if len(variants) == 1:
                label_mapping[variants[0]] = prefix
            else:
                # 有多个变体（比如 A_1 和 A_2）
                context_sets = [frozenset(label_to_contexts[v]) for v in variants]
                if len(set(context_sets)) == 1:
                    # 指向的系统完全相同且（M/A/F/N）也完全一模一样
                    for v in variants:
                        label_mapping[v] = prefix
                else:
                    # 变体之间存在分歧
                    protected_prefixes.add(prefix)
                    for v in variants:
                        label_mapping[v] = v
        self.label_mapping = label_mapping
        self.protected_prefixes = protected_prefixes


    def get_ultimate_label(self, raw_label):
        if raw_label in ['other', 'pad']:
            return raw_label
        
        # 如果在映射表里，直接用
        if raw_label in self.label_mapping:
            return self.label_mapping[raw_label]
            
        # 如果数据集里有，但规则表里没显式写出：
        prefix = re.sub(r'_[0-9]+$', '', raw_label)
        
        if prefix in self.protected_prefixes:
            return raw_label  
        else:
            return prefix     
    
    def __call__(self, raw_label):
        return self.get_ultimate_label(raw_label)