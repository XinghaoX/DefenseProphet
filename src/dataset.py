import torch
from torch.utils.data import Dataset
import pickle as pk


class PKLDataset(Dataset):
    def __init__(self, pkl_path, window_size=64, stride=32, is_same_contig=False):
        self.window_size = window_size
        self.stride = stride
        
        self.is_same_contig = is_same_contig
        with open(pkl_path, 'rb') as f:
            data = pk.load(f)
        self.embeddings = data['embeddings']
        self.gene_ids = data['gene_id']
        self.descriptions = data['description']
        self._init_windows()

    def _init_windows(self):
        self.num_proteins = len(self.gene_ids)
        self.windows = []
        
        if self.num_proteins == 0:
            return

        contig_ranges = []

        if self.is_same_contig:
            contig_ranges.append((0, self.num_proteins))
        else:
            current_contig = None
            contig_start = 0

            for i, gene_id in enumerate(self.gene_ids):
                # 提取 Contig ID 
                contig_id = gene_id.rsplit('_', 1)[0]
                
                if current_contig is None:
                    current_contig = contig_id
                    contig_start = i
                elif contig_id != current_contig:
                    contig_ranges.append((contig_start, i))
                    current_contig = contig_id
                    contig_start = i
                    
            contig_ranges.append((contig_start, self.num_proteins))

        # 在每个 Contig 内部独立生成滑窗
        for start_idx, end_idx in contig_ranges:
            contig_len = end_idx - start_idx
            
            if contig_len <= self.window_size:
                # pos_type = 0 表示这是该 Contig 的唯一窗口
                self.windows.append((start_idx, end_idx, 0)) 
            else:
                c_windows = []
                for w_start in range(start_idx, end_idx, self.stride):
                    w_end = min(w_start + self.window_size, end_idx)
                    c_windows.append((w_start, w_end))
                    if w_end == end_idx:
                        break
                
                num_c_windows = len(c_windows)
                for i, (w_start, w_end) in enumerate(c_windows):
                    if i == 0:
                        pos_type = 1  # 1: 该 Contig 的首个窗口
                    elif i == num_c_windows - 1:
                        pos_type = 3  # 3: 该 Contig 的末尾窗口
                    else:
                        pos_type = 2  # 2: 该 Contig 的中间窗口
                        
                    self.windows.append((w_start, w_end, pos_type))

    def __len__(self): 
        return len(self.windows)

    def __getitem__(self, idx):
        start, end, pos_type = self.windows[idx]
        emb = self.embeddings[start:end]
        actual_len = end - start
        
        emb_tensor = torch.tensor(emb, dtype=torch.float32)
        attention_mask = torch.ones(self.window_size, dtype=torch.long)
        
        if actual_len < self.window_size:
            pad_len = self.window_size - actual_len
            pad_tensor = torch.zeros((pad_len, emb_tensor.shape[1]), dtype=torch.float32)
            emb_tensor = torch.cat([emb_tensor, pad_tensor], dim=0)
            attention_mask[actual_len:] = 0 
            
        return {
            "inputs_embeds": emb_tensor, 
            "attention_mask": attention_mask,
            "pos_type": pos_type,      
            "actual_len": actual_len
        }

