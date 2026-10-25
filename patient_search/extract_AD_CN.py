import os
import pandas as pd

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

print("正在读取各种临床量表 CSV 文件...")
df_dx = pd.read_csv(os.path.join(script_dir, 'All_Subjects_DXSUM_19Jan2026.csv'))
df_demo = pd.read_csv(os.path.join(script_dir, 'All_Subjects_PTDEMOG_19Jan2026.csv'))
df_apoe = pd.read_csv(os.path.join(script_dir, 'APOERES_20Jan2026.csv'))
df_mmse = pd.read_csv(os.path.join(script_dir, 'MMSE_20Jan2026.csv'))
df_cdr = pd.read_csv(os.path.join(script_dir, 'CDR_20Jan2026.csv'))

# ==========================================
# 1. 提取基线 AD 和 CN
# ==========================================
def parse_month(viscode2):
    v = str(viscode2).lower().strip()
    if v in ['bl', 'baseline', 'sc']: return 0
    if v.startswith('m'):
        try: return int(v[1:])
        except: return None
    return None

df_dx['Month'] = df_dx['VISCODE2'].apply(parse_month)
baseline_df = df_dx[df_dx['Month'] == 0].copy()

if 'EXAMDATE' in baseline_df.columns:
    baseline_df['EXAMDATE'] = pd.to_datetime(baseline_df['EXAMDATE'], errors='coerce')
    baseline_df = baseline_df.sort_values('EXAMDATE')
baseline_df = baseline_df.drop_duplicates(subset=['PTID'], keep='first')
baseline_df.rename(columns={'EXAMDATE': 'BaselineDate'}, inplace=True)

# 核心：CN 赋予标签 2，AD 赋予标签 3
cn_base = baseline_df[baseline_df['DIAGNOSIS'] == 1].copy()
cn_base['Group'] = 'CN'
cn_base['Label'] = 2  

ad_base = baseline_df[baseline_df['DIAGNOSIS'] == 3].copy()
ad_base['Group'] = 'AD'
ad_base['Label'] = 3  

combined_base = pd.concat([cn_base, ad_base])

# ==========================================
# 2. 合并多模态特征 (复用你原本非常严谨的合并逻辑)
# ==========================================
def get_unique_baseline(df_source, cols_to_keep):
    baseline_codes = ['sc', 'bl', 'v01', 'v02']
    mask = df_source['VISCODE'].isin(baseline_codes)
    df_subset = df_source[mask].copy()
    if df_subset.empty: return pd.DataFrame(columns=['PTID'] + cols_to_keep)
    priority_map = {'sc': 0, 'bl': 1, 'v01': 2, 'v02': 3}
    df_subset['priority'] = df_subset['VISCODE'].map(priority_map)
    df_unique = df_subset.sort_values('priority').drop_duplicates(subset=['PTID'], keep='first')
    return df_unique[['PTID'] + cols_to_keep]

df_demo_unique = get_unique_baseline(df_demo, ['PTGENDER', 'PTDOB', 'PTEDUCAT'])
df_feat = combined_base[['PTID', 'Group', 'Label', 'BaselineDate']].merge(df_demo_unique, on='PTID', how='inner')

def calc_age(row):
    try:
        birth = pd.to_datetime(row['PTDOB'], errors='coerce') 
        exam = pd.to_datetime(row['BaselineDate'], errors='coerce')
        if pd.isna(birth) or pd.isna(exam): return None
        return round((exam - birth).days / 365.25, 1)
    except: return None

df_feat['Age'] = df_feat.apply(calc_age, axis=1)
df_feat['Sex_Binary'] = df_feat['PTGENDER'].apply(lambda x: 0 if x == 1 else 1)

df_apoe_cln = df_apoe.dropna(subset=['GENOTYPE']).drop_duplicates(subset=['PTID'])
def get_apoe_status(geno):
    if pd.isna(geno): return None
    if '4' in str(geno): return 1
    return 0
df_apoe_cln['ApoE4_Status'] = df_apoe_cln['GENOTYPE'].apply(get_apoe_status)
df_feat = df_feat.merge(df_apoe_cln[['PTID', 'ApoE4_Status']], on='PTID', how='inner')

df_mmse_unique = get_unique_baseline(df_mmse, ['MMSCORE'])
df_cdr_unique = get_unique_baseline(df_cdr, ['CDRSB'])
df_feat = df_feat.merge(df_mmse_unique, on='PTID', how='inner')
df_feat = df_feat.merge(df_cdr_unique, on='PTID', how='inner')

final_df = df_feat[[
    'PTID', 'Group', 'Label', 
    'Age', 'Sex_Binary', 'PTEDUCAT', 
    'ApoE4_Status', 
    'MMSCORE', 'CDRSB'
]]
final_df.columns = ['SubjectID', 'Group', 'Label', 'Age', 'Sex', 'Education', 'ApoE4', 'MMSE', 'CDR_SB']
final_df = final_df.dropna()

# ==========================================
# 3. 分别提取、正则排序并保存 CSV 和 TXT
# ==========================================
print("-" * 30)
for group_name in ['CN', 'AD']:
    # 提取单组
    group_df = final_df[final_df['Group'] == group_name].copy()
    
    # 正则排序 (按照 002_S_xxxx 的顺序)
    group_df['prefix_numeric'] = group_df['SubjectID'].str.extract(r'(\d+)_S_').astype(int)
    group_df['suffix_numeric'] = group_df['SubjectID'].str.extract(r'_S_(\d+)').astype(int)
    group_sorted = group_df.sort_values(by=['prefix_numeric','suffix_numeric'])
    group_sorted.drop(columns=['prefix_numeric','suffix_numeric'], inplace=True)
    group_sorted = group_sorted.reset_index(drop=True)
    
    # 1. 导出 CSV
    csv_name = f'{group_name}_sorted.csv'
    group_sorted.to_csv(os.path.join(script_dir, csv_name), index=False)
    
    # 2. 导出 TXT
    txt_name = f'{group_name}_subjects.txt'
    with open(os.path.join(script_dir, txt_name), 'w') as f:
        f.write(",".join(group_sorted['SubjectID'].tolist()))
        
    print(f"✅ {group_name} 数据处理完毕！有效人数: {len(group_sorted)} 例。")
    print(f"   -> 已生成表: {csv_name}")
    print(f"   -> 已生成名单: {txt_name}\n")
print("-" * 30)