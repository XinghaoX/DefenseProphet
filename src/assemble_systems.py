import pandas as pd

class DefenseAssembler:
    def __init__(self, rules_file):
        self.rules_df = pd.read_csv(rules_file, sep='\t')
        self.rules_dict = {}

        for _, row in self.rules_df.iterrows():
            subsys = row['Subsystem']
            
            mand_str = str(row['Mandatory'])
            acc_str = str(row['Accessory'])
            forb_str = str(row['Forbidden'])
            neu_str = str(row['Neutral']) 
            
            mand = set([x.strip() for x in mand_str.split(',')] if mand_str != 'nan' else [])
            acc = set([x.strip() for x in acc_str.split(',')] if acc_str != 'nan' else [])
            forb = set([x.strip() for x in forb_str.split(',')] if forb_str != 'nan' else [])
            neu = set([x.strip() for x in neu_str.split(',')] if neu_str != 'nan' else [])
            
            self.rules_dict[subsys] = {
                'System': row['System'],
                'Min_mandatory': row['Min_mandatory'],
                'Minimum_genes': row['Minimum_genes'],
                'Mandatory': mand,
                'Accessory': acc,
                'Forbidden': forb,
                'Neutral': neu
            }
            
    def _get_gene_idx(self, gene_id):
        try:
            return int(gene_id.split('_')[-1])
        except ValueError:
            return 0

    def assemble(self, predictions_csv, max_gap=5):
        df_full = pd.read_csv(predictions_csv)

        df_full['Assembly_Status'] = 'Background' 
        df_full['Assigned_System'] = None
        df_full['Assigned_Subsystem'] = None
        df_full['Assigned_Role_Detail'] = None

        defense_mask = (df_full['Multi_Predicted_Class'] != 'other') & (df_full['Binary_Predicted_Class'] == 'defense')
        df_full.loc[defense_mask, 'Assembly_Status'] = 'Unassembled'
        
        df_def = df_full[defense_mask].copy()
        if df_def.empty:
            return pd.DataFrame(), df_full
            
        df_def['gene_idx'] = df_def['Gene_ID'].apply(self._get_gene_idx)
        
        results = []
        gene_annotations = {}

        for contig, group in df_def.groupby('Contig_ID'):
            group = group.sort_values('gene_idx')
            clusters = []
            current_cluster = []
            for _, row in group.iterrows():
                if not current_cluster:
                    current_cluster.append(row)
                else:
                    last_gene_idx = current_cluster[-1]['gene_idx']
                    if row['gene_idx'] - last_gene_idx <= max_gap:
                        current_cluster.append(row)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [row]
            if current_cluster:
                clusters.append(current_cluster)

            for cluster in clusters:
                cluster_genes = [row['Multi_Predicted_Class'] for row in cluster]
                
                for subsys, rule in self.rules_dict.items():
                    mand_hits = [g for g in cluster_genes if g in rule['Mandatory']]
                    acc_hits = [g for g in cluster_genes if g in rule['Accessory']]
                    forb_hits = [g for g in cluster_genes if g in rule['Forbidden']]
                    neutral_hits = [g for g in cluster_genes if g in rule['Neutral']]
                    
                    if forb_hits:
                        continue 
                        
                    unique_mand_count = len(set(mand_hits))
                    total_hits = len(mand_hits) + len(acc_hits)
                    
                    if unique_mand_count >= rule['Min_mandatory'] and total_hits >= rule['Minimum_genes']:
                        
                        valid_labels = rule['Mandatory'].union(rule['Accessory']).union(rule['Neutral'])
                        valid_genes_in_cluster = [row for row in cluster if row['Multi_Predicted_Class'] in valid_labels]
                        
                        start_gene = valid_genes_in_cluster[0]['Gene_ID']
                        end_gene = valid_genes_in_cluster[-1]['Gene_ID']
                        
                        results.append({
                            'Contig_ID': contig,
                            'System': rule['System'],
                            'Subsystem': subsys,
                            'Start_Gene': start_gene,
                            'End_Gene': end_gene,
                            'Total_Genes_Found': total_hits,
                            'Unique_Mandatory_Found': unique_mand_count,
                            'Mandatory_Hits': ','.join(set(mand_hits)),
                            'Accessory_Hits': ','.join(set(acc_hits)),
                            'Neutral_Hits_Found': ','.join(set(neutral_hits)) if neutral_hits else 'None'
                        })

                        for row in valid_genes_in_cluster:
                            gid = row['Gene_ID']
                            gene_class = row['Multi_Predicted_Class']
                            if gene_class in rule['Mandatory']:
                                role = 'Mandatory'
                            elif gene_class in rule['Accessory']:
                                role = 'Accessory'
                            elif gene_class in rule['Neutral']:
                                role = 'Neutral'
                            else:
                                role = 'Unknown'

                            role_detail = f"{subsys}:{role}"

                            if gid in gene_annotations:
                                gene_annotations[gid]['System'].add(rule['System'])
                                gene_annotations[gid]['Subsystem'].add(subsys)
                                gene_annotations[gid]['Role_Detail'].add(role_detail)
                            else:
                                # 第一次记录时，初始化为集合 (set)
                                gene_annotations[gid] = {
                                    'System': {rule['System']}, 
                                    'Subsystem': {subsys},
                                    'Role_Detail': {role_detail}
                                }

        # 映射回基因全景表
        if gene_annotations:
            # 提前拼接好字符串
            system_map = {gid: ' | '.join(sorted(list(ann['System']))) for gid, ann in gene_annotations.items()}
            subsystem_map = {gid: ' | '.join(sorted(list(ann['Subsystem']))) for gid, ann in gene_annotations.items()}
            role_map = {gid: ' | '.join(sorted(list(ann['Role_Detail']))) for gid, ann in gene_annotations.items()}
            
            # 批量写入 DataFrame
            df_full['Assigned_System'] = df_full['Gene_ID'].map(system_map).fillna(df_full['Assigned_System'])
            df_full['Assigned_Subsystem'] = df_full['Gene_ID'].map(subsystem_map).fillna(df_full['Assigned_Subsystem'])
            df_full['Assigned_Role_Detail'] = df_full['Gene_ID'].map(role_map).fillna(df_full['Assigned_Role_Detail'])
            
            # 更新状态
            df_full.loc[df_full['Assigned_System'].notna(), 'Assembly_Status'] = 'Assembled'

        df_results = pd.DataFrame(results)
        
        return df_results, df_full

if __name__ == "__main__":
    RULES_PATH = './DefenseFinder_rules_complement.tsv'
    PREDICTIONS_PATH = './GB_GCA_000008085.1_protein_predictions.csv'
    OUTPUT_SUMMARY_PATH = './Assembled_Systems_Summary.csv'
    OUTPUT_GENES_PATH = './Annotated_Genes_Full.csv' # 新增的全局结果文件
    
    assembler = DefenseAssembler(RULES_PATH)
    
    # 接收两个返回值
    df_systems, df_full_genes = assembler.assemble(PREDICTIONS_PATH, max_gap=5)
    
    if not df_systems.empty:
        df_systems.to_csv(OUTPUT_SUMMARY_PATH, index=False)
        
    # 保存全景表（无论有没有组装成功都保存，方便你查看所有的孤儿基因）
    df_full_genes.to_csv(OUTPUT_GENES_PATH, index=False)
    print(f"Full Annotated Genes saved to: {OUTPUT_GENES_PATH}")
    
    # 打印一些有趣的统计
    orphan_count = len(df_full_genes[df_full_genes['Assembly_Status'] == 'Orphan (Unassembled)'])
    print(f"\nQuick Stats:")
    print(f"   - Complete Systems Found: {len(df_systems)}")
    print(f"   - Orphan (Unassembled) Defense Genes: {orphan_count}")