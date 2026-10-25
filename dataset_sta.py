import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

def correct_labels_and_generate_table(clinical_csv_path, correct_labels_csv_path):
    print("1. 正在读取并校正数据...\n")
    # 读取两张表
    df_clinical = pd.read_csv(clinical_csv_path)
    df_correct = pd.read_csv(correct_labels_csv_path)
    
    # 统一列名：将 all_data.csv 里的 'subject_id' 改为 'SubjectID' 以便对齐
    if 'Subject_ID' in df_correct.columns:
        df_correct = df_correct.rename(columns={'Subject_ID': 'SubjectID'})
        
    # 剔除旧表里不准确的分类列（Label, Group, Class等），只保留特征列
    cols_to_drop = [c for c in ['Label', 'Group', 'Class'] if c in df_clinical.columns]
    df_features = df_clinical.drop(columns=cols_to_drop)
    
    # 【核心逻辑】：以内连接 (inner join) 合并数据
    # 这样只保留 all_data.csv 里有的人，并且使用它里面的权威 Label
    df = pd.merge(df_correct[['SubjectID', 'Label']], df_features, on='SubjectID', how='inner')
    
    # 检查清洗后的数据量
    df_smci = df[df['Label'] == 0]  #sMCI
    df_pmci = df[df['Label'] == 1]  #pMCI
    n_smci = len(df_smci)
    n_pmci = len(df_pmci)
    print(f"✅ 数据合并完成！最终用于统计的总人数: {len(df)}")
    print(f"   - sMCI 人数: {n_smci}")
    print(f"   - pMCI 人数: {n_pmci}\n")

    print("2. 正在计算统计学差异并生成 Table 1...\n")
    # 定义连续变量和分类变量
    continuous_vars = {
        'Age': 'Age (years)',
        'Education': 'Education (years)',
        'MMSE': 'MMSE Score',
        'CDR_SB': 'CDR-SB Score'
    }
    
    categorical_vars = {
        'Sex': 'Sex (Female %)', 
        'ApoE4': 'ApoE4 Carrier (%)'
    }
    
    table_rows = []

    # --- 处理连续变量 ---
    for col, display_name in continuous_vars.items():
        mean_s, std_s = df_smci[col].mean(), df_smci[col].std()
        mean_p, std_p = df_pmci[col].mean(), df_pmci[col].std()
        
        # 独立样本 t 检验
        t_stat, p_val = ttest_ind(df_smci[col].dropna(), df_pmci[col].dropna(), equal_var=False)
        
        table_rows.append({
            'Characteristics': display_name,
            f'sMCI (N={n_smci})': f"{mean_s:.2f} ± {std_s:.2f}",
            f'pMCI (N={n_pmci})': f"{mean_p:.2f} ± {std_p:.2f}",
            'p-value': "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        })

    # --- 处理分类变量 ---
    for col, display_name in categorical_vars.items():
        count_s = df_smci[col].sum()
        perc_s = (count_s / n_smci) * 100 if n_smci > 0 else 0
        
        count_p = df_pmci[col].sum()
        perc_p = (count_p / n_pmci) * 100 if n_pmci > 0 else 0
        
        # 卡方检验
        contingency_table = pd.crosstab(df[col], df['Label'])
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        
        table_rows.append({
            'Characteristics': display_name,
            f'sMCI (N={n_smci})': f"{int(count_s)} ({perc_s:.1f}%)",
            f'pMCI (N={n_pmci})': f"{int(count_p)} ({perc_p:.1f}%)",
            'p-value': "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        })

    # 生成 DataFrame
    table1_df = pd.DataFrame(table_rows)
    
    # 打印并在同目录输出新的表
    print("-" * 75)
    print("Table 1: Baseline Demographics and Clinical Characteristics (Corrected)")
    print("-" * 75)
    print(table1_df.to_string(index=False))
    print("-" * 75)
    
    out_file = 'Table1_Baseline_Characteristics_Corrected.csv'
    table1_df.to_csv(out_file, index=False)
    
    # 顺手把带有正确标签的完整特征表也保存下来，以后写 DataLoader 时直接用它
    df.to_csv("patient_search/corrected_final_clinical_dataset.csv", index=False)
    print(f"\n✅ 统计表格已保存为: {out_file}")
    print(f"✅ 更新了正确标签的最终特征数据集已保存为: patient_search/corrected_final_clinical_dataset.csv")

if __name__ == "__main__":
    # 请确认这两个路径与你电脑里的实际位置对应
    clinical_csv_path = "patient_search/final_clinical_dataset.csv" 
    correct_labels_csv_path = "data/clinical/MCI/all_data.csv" 
    
    correct_labels_and_generate_table(clinical_csv_path, correct_labels_csv_path)