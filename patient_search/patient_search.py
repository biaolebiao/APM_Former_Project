import os
import pandas as pd
import numpy as np

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 读取数据
# ==========================================
print("正在读取 CSV 文件...")
df_dx = pd.read_csv(os.path.join(script_dir, 'All_Subjects_DXSUM_19Jan2026.csv'))
df_demo = pd.read_csv(os.path.join(script_dir, 'All_Subjects_PTDEMOG_19Jan2026.csv'))
df_apoe = pd.read_csv(os.path.join(script_dir, 'APOERES_20Jan2026.csv'))
df_mmse = pd.read_csv(os.path.join(script_dir, 'MMSE_20Jan2026.csv'))
df_cdr = pd.read_csv(os.path.join(script_dir, 'CDR_20Jan2026.csv'))

# ==========================================
# 2. 标签生成 (sMCI/pMCI)
# ==========================================
def parse_month(viscode2):
    v = str(viscode2).lower().strip()
    if v in ['bl', 'baseline', 'sc']: return 0
    if v.startswith('m'):
        try: return int(v[1:])
        except: return None
    return None

df_dx['Month'] = df_dx['VISCODE2'].apply(parse_month)

# 筛选基线 MCI
base_mci = df_dx[(df_dx['Month'] == 0) & (df_dx['DIAGNOSIS'] == 2)][['PTID', 'EXAMDATE']].copy()
base_mci.rename(columns={'EXAMDATE': 'BaselineDate'}, inplace=True)
# 确保基线 MCI 列表唯一
base_mci = base_mci.sort_values('BaselineDate').drop_duplicates(subset=['PTID'], keep='first')
mci_ptids = base_mci['PTID'].unique()

print(f"--- 步骤1: 基线 MCI 总人数: {len(mci_ptids)} ---")

labels = []
for ptid in mci_ptids:
    records = df_dx[df_dx['PTID'] == ptid]
    followup_36 = records[(records['Month'] > 0) & (records['Month'] <= 36)]
    
    has_converted = False
    if not followup_36.empty:
        if 3 in followup_36['DIAGNOSIS'].values:
            has_converted = True
            
    if has_converted:
        labels.append({'PTID': ptid, 'Label': 'pMCI', 'Class': 1})
    else:
        last_visit = records['Month'].max()
        if last_visit >= 36:
            labels.append({'PTID': ptid, 'Label': 'sMCI', 'Class': 0})

df_labels = pd.DataFrame(labels)
print(f"--- 步骤2: 标签筛选结果 (sMCI+pMCI): {len(df_labels)} ---")
# 检查这里是否有重复
if df_labels['PTID'].duplicated().any():
    print("⚠️ 警告: df_labels 中存在重复 ID，正在去重...")
    df_labels = df_labels.drop_duplicates(subset=['PTID'])

# ==========================================
# 3. 多模态特征合并 (防止膨胀的核心)
# ==========================================

def get_unique_baseline(df_source, cols_to_keep):
    """
    智能提取唯一的基线数据
    优先级: sc > bl > v01 > v02
    """
    # 1. 筛选可能的基线行
    baseline_codes = ['sc', 'bl', 'v01', 'v02']
    mask = df_source['VISCODE'].isin(baseline_codes)
    df_subset = df_source[mask].copy()
    
    # 2. 如果没有任何匹配，返回空表
    if df_subset.empty:
        return pd.DataFrame(columns=['PTID'] + cols_to_keep)

    # 3. 定义优先级 (数值越小优先级越高)
    priority_map = {'sc': 0, 'bl': 1, 'v01': 2, 'v02': 3}
    df_subset['priority'] = df_subset['VISCODE'].map(priority_map)
    
    # 4. 关键步骤：排序并去重！确保每个 PTID 只留一行
    # 这样 merge 的时候就是 1对1，不会膨胀
    df_unique = df_subset.sort_values('priority').drop_duplicates(subset=['PTID'], keep='first')
    
    return df_unique[['PTID'] + cols_to_keep]

# --- 3.1 合并人口学 ---
df_demo_unique = get_unique_baseline(df_demo, ['PTGENDER', 'PTDOB', 'PTEDUCAT'])
# 先把基线日期合并回来
df_feat = df_labels.merge(base_mci, on='PTID', how='left') 
# 合并人口学 (注意使用 inner join 会自动过滤掉没有人口学的人，但绝不应增加人数)
df_feat = df_feat.merge(df_demo_unique, on='PTID', how='inner')

print(f"--- 步骤3: 合并人口学后人数: {len(df_feat)} (正常应 <= 步骤2人数) ---")

# --- 3.2 计算年龄 ---
def calc_age(row):
    try:
        birth = pd.to_datetime(row['PTDOB'], errors='coerce') 
        exam = pd.to_datetime(row['BaselineDate'], errors='coerce')
        if pd.isna(birth) or pd.isna(exam): return None
        return round((exam - birth).days / 365.25, 1)
    except:
        return None

df_feat['Age'] = df_feat.apply(calc_age, axis=1)
df_feat['Sex_Binary'] = df_feat['PTGENDER'].apply(lambda x: 0 if x == 1 else 1)

# --- 3.3 合并 ApoE ---
df_apoe_cln = df_apoe.dropna(subset=['GENOTYPE']).drop_duplicates(subset=['PTID'])
def get_apoe_status(geno):
    if pd.isna(geno): return None
    if '4' in str(geno): return 1
    return 0
df_apoe_cln['ApoE4_Status'] = df_apoe_cln['GENOTYPE'].apply(get_apoe_status)

df_feat = df_feat.merge(df_apoe_cln[['PTID', 'ApoE4_Status']], on='PTID', how='inner')
print(f"--- 步骤4: 合并 ApoE 后人数: {len(df_feat)} ---")

# --- 3.4 合并 MMSE & CDR ---
df_mmse_unique = get_unique_baseline(df_mmse, ['MMSCORE'])
df_cdr_unique = get_unique_baseline(df_cdr, ['CDRSB'])

df_feat = df_feat.merge(df_mmse_unique, on='PTID', how='inner')
df_feat = df_feat.merge(df_cdr_unique, on='PTID', how='inner')

print(f"--- 步骤5: 合并 MMSE/CDR 后人数: {len(df_feat)} ---")

# ==========================================
# 4. 导出最终表
# ==========================================
final_df = df_feat[[
    'PTID', 'Label', 'Class', 
    'Age', 'Sex_Binary', 'PTEDUCAT', 
    'ApoE4_Status', 
    'MMSCORE', 'CDRSB'
]]
final_df.columns = ['SubjectID', 'Group', 'Label', 'Age', 'Sex', 'Education', 'ApoE4', 'MMSE', 'CDR_SB']

final_df = final_df.dropna()

print("-" * 30)
print(f"✅ 最终数据集: {len(final_df)} 例")
print(f"sMCI: {len(final_df[final_df['Label']=='sMCI'])}")
print(f"pMCI: {len(final_df[final_df['Label']=='pMCI'])}")
print("-" * 30)

output_path = os.path.join(script_dir, 'final_clinical_dataset.csv')
final_df.to_csv(output_path, index=False)
print(f"文件已保存为: {output_path}")