import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from .dataset import PKLDataset # H5Dataset
import pandas as pd
import numpy as np
import pickle as pk
from .utils import seed_everything
seed_everything(0)
ERROR_LOG_PATH = './error_log.txt'

def fast_read_fasta(file_path):
    gene_ids, descriptions, sequences = [], [], []
    with open(file_path, 'r') as f:
        seq_chunks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq_chunks:
                    sequences.append("".join(seq_chunks).replace("*", ""))
                    seq_chunks = []
                header = line[1:]
                parts = header.split(" ", 1)
                gene_ids.append(parts[0])
                descriptions.append(parts[1] if len(parts) > 1 else "")
            else:
                seq_chunks.append(line)
        if seq_chunks:
            sequences.append("".join(seq_chunks).replace("*", ""))
            
    return gene_ids, descriptions, sequences


def get_dynamic_batch_indices(sequences, toks_per_batch, max_seq_len=1022):
    sizes = [(min(len(s), max_seq_len), i) for i, s in enumerate(sequences)]
    sizes.sort() 
    
    batches = []
    buf = []
    max_len = 0

    for sz, i in sizes:
        if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
            if buf:
                batches.append(buf)
            buf = []
            max_len = 0
        max_len = max(max_len, sz)
        buf.append(i)

    if buf:
        batches.append(buf)
    return batches

def encode_fasta_to_pkl_dynamic(fasta_path, output_pkl_path, model, tokenizer, device, toks_per_batch=4096):
    if os.path.exists(output_pkl_path):
        return output_pkl_path
        
    try:
        gene_ids, descriptions, sequences = fast_read_fasta(fasta_path)
    except Exception as e:
        with open(ERROR_LOG_PATH, 'a') as f:
            f.write(f"{fasta_path}\tReadError: {e}\n")
        return None

    if not sequences:
        return None

    num_seqs = len(sequences)
    hidden_size = model.config.hidden_size 

    final_embeddings = np.zeros((num_seqs, hidden_size), dtype=np.float32)

    batches = get_dynamic_batch_indices(sequences, toks_per_batch=toks_per_batch)
    success = True
    for batch_indices in batches:
        batch_seqs = [sequences[i] for i in batch_indices]
        
        try:
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding='longest', truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embedding_repr = outputs.last_hidden_state
                
                attention_mask = inputs['attention_mask']
                aa_mask = attention_mask.clone()
                aa_mask[:, 0] = 0  # 忽略 <cls>
                
                seq_lengths = attention_mask.sum(dim=1).long()
                # for b_idx in range(len(seq_lengths)):
                #     if seq_lengths[b_idx] > 1:
                #         aa_mask[b_idx, seq_lengths[b_idx] - 1] = 0  # 忽略 <eos>
                valid_idx = seq_lengths > 1
                row_idx = torch.arange(attention_mask.size(0), device=device)
                aa_mask[row_idx[valid_idx], seq_lengths[valid_idx] - 1] = 0
                        
                aa_mask_expanded = aa_mask.unsqueeze(-1).expand(embedding_repr.size())
                masked_embedding_repr = embedding_repr * aa_mask_expanded
                
                sum_embedding_repr = masked_embedding_repr.sum(dim=1)
                non_zero_count = aa_mask_expanded.sum(dim=1)
                non_zero_count = torch.clamp(non_zero_count, min=1e-9)
                mean_embeddings = sum_embedding_repr / non_zero_count
            
            batch_embeddings = mean_embeddings.cpu().numpy()
            for b_idx, orig_idx in enumerate(batch_indices):
                final_embeddings[orig_idx] = batch_embeddings[b_idx]
                
        except Exception as e:
            with open(ERROR_LOG_PATH, 'a') as f:
                f.write(f"{fasta_path}\tBatchError: {e}\n")
            success = False
            break

    if not success:
        return None
    
    save_data = {
        'embeddings': final_embeddings,     # shape: (N, hidden_size)
        'gene_id': gene_ids,                # len: N
        'description': descriptions         # len: N
    }

    tmp_path = output_pkl_path + ".tmp"
    with open(tmp_path, 'wb') as f:
        pk.dump(save_data, f)
    os.rename(tmp_path, output_pkl_path)
    
    return output_pkl_path

# def inference_collate_fn(batch):
#     inputs_embeds = torch.stack([item['inputs_embeds'] for item in batch])
#     gene_ids = [item['gene_id'] for item in batch]
#     return {
#         "inputs_embeds": inputs_embeds,
#         "gene_id": gene_ids
#     }

def run_pipeline(files_to_process: list, binary_model, multi_model, output_csv_dir, id2label, binary_id2label,
                window_size=64, stride=32, device="cuda:0", is_same_contig= False):
    '''
    files_to_process: 需要推理的编码后的文件的路径列表
    binary_model: 二分类模型
    multi_model: 多分类模型
    '''
    binary_model = binary_model.to(device)
    binary_model.eval()
    
    multi_model = multi_model.to(device)
    multi_model.eval()
    
    os.makedirs(output_csv_dir, exist_ok=True)
        
    cut = (window_size - stride) // 2 

    for file_path in tqdm(files_to_process):
        file_name = os.path.basename(file_path)
        genome_id = file_name.replace('.h5', '').replace('.pkl', '')
        output_csv_path = os.path.join(output_csv_dir, f"{genome_id}_predictions.csv")
        
        if os.path.exists(output_csv_path):
            continue
            
        # if file_path.endswith('.h5'):
        #     dataset = H5Dataset(file_path, window_size, stride)
        if file_path.endswith('.pkl'):
            dataset = PKLDataset(file_path, window_size, stride, is_same_contig=is_same_contig)
        else:
            print(f"不支持的文件: {file_name}")
            continue
            
        global_gene_ids = dataset.gene_ids 
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        global_descriptions = dataset.descriptions

        all_binary_preds = []
        all_binary_confs = []
        all_multi_preds = []
        all_multi_confs = []
        
        with torch.no_grad():
            for batch in dataloader:
                embeds = batch['inputs_embeds'].to(device)
                masks = batch['attention_mask'].to(device)
                
                pos_types = batch['pos_type'].numpy()
                actual_lens = batch['actual_len'].numpy()
                
                # 二分类预测
                bin_outputs = binary_model(inputs_embeds=embeds, attention_mask=masks)
                bin_logits = bin_outputs.logits
                bin_probs = torch.softmax(bin_logits, dim=-1)
                bin_conf, bin_pred = torch.max(bin_probs, dim=-1)
                
                # 多分类预测
                mul_outputs = multi_model(inputs_embeds=embeds, attention_mask=masks)
                mul_logits = mul_outputs.logits
                mul_probs = torch.softmax(mul_logits, dim=-1)
                mul_conf, mul_pred = torch.max(mul_probs, dim=-1)
                
                for i in range(len(pos_types)):
                    p_type = pos_types[i]
                    length = actual_lens[i]
                    
                    b_p = bin_pred[i].cpu().numpy()
                    b_c = bin_conf[i].cpu().numpy()
                    m_p = mul_pred[i].cpu().numpy()
                    m_c = mul_conf[i].cpu().numpy()
                    
                    if p_type == 0:
                        # Contig 只有一个窗口
                        keep_slice = slice(0, length)
                    elif p_type == 1:
                        # Contig 的开头窗口
                        keep_slice = slice(0, cut + stride) 
                    elif p_type == 3:
                        # Contig 的结尾窗口
                        keep_slice = slice(cut, length)
                    else:
                        # Contig 的中间窗口 (p_type == 2)
                        keep_slice = slice(cut, cut + stride)
                        
                    all_binary_preds.extend(b_p[keep_slice].tolist())
                    all_binary_confs.extend(b_c[keep_slice].tolist())
                    all_multi_preds.extend(m_p[keep_slice].tolist())
                    all_multi_confs.extend(m_c[keep_slice].tolist())
                    
        assert len(all_binary_preds) == len(global_gene_ids), f"二分类长度对齐错误！"
        assert len(all_multi_preds) == len(global_gene_ids), f"多分类长度对齐错误！"
        
        if binary_id2label is not None:
            all_binary_preds = [binary_id2label.get(p, p) for p in all_binary_preds]
            
        if id2label is not None:
            all_multi_preds = [id2label.get(p, p) for p in all_multi_preds]

        contig_ids = [gid.rsplit('_', 1)[0] for gid in global_gene_ids]

        df = pd.DataFrame({
            "Contig_ID": contig_ids,
            "Gene_ID": global_gene_ids,
            "Binary_Predicted_Class": all_binary_preds,
            "Binary_Confidence": all_binary_confs,
            "Multi_Predicted_Class": all_multi_preds,
            "Multi_Confidence": all_multi_confs,
            "Description": global_descriptions
        })
        df.to_csv(output_csv_path, index=False)