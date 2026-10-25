import os
import subprocess

def batch_convert_dicom_to_nifti(raw_data_root, nifti_root):
    """
    批量将整个数据集的 DICOM 转换为 NIfTI。
    假设 raw_data_root 下直接是各个患者的文件夹（如 011_S_0002, 011_S_0003...）
    """
    os.makedirs(nifti_root, exist_ok=True)
    dcm2niix_path = "dcm2niix" # 确保路径正确
    
    # 获取根目录下所有的患者文件夹名称
    patient_folders = [f for f in os.listdir(raw_data_root) 
                       if os.path.isdir(os.path.join(raw_data_root, f))]
    
    total_patients = len(patient_folders)
    print(f"总共找到 {total_patients} 个患者文件夹，开始批量转换...")
    
    success_count = 0
    fail_count = 0
    
    for i, patient_id in enumerate(patient_folders, 1):
        input_dir = os.path.join(raw_data_root, patient_id)
        # 为每个患者在 NIfTI 目录下创建一个专属文件夹
        output_dir = os.path.join(nifti_root, patient_id)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[{i}/{total_patients}] 正在处理: {patient_id} ...", end=" ")
        
        command = [
            dcm2niix_path,
            "-z", "y",
            "-f", "%i_%p", # 文件名: ID_序列名
            "-o", output_dir,
            input_dir
        ]
        
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print("✅ 成功")
                success_count += 1
            else:
                print("❌ 失败")
                fail_count += 1
                # 可选：将失败信息写入日志文件
                with open("conversion_errors.log", "a") as f:
                    f.write(f"--- Error for {patient_id} ---\n{result.stderr}\n")
        except Exception as e:
            print(f"❌ 发生异常: {e}")
            fail_count += 1

    print("-" * 30)
    print(f"批量转换结束！成功: {success_count}，失败: {fail_count}")
    print(f"NIfTI 数据已保存在: {nifti_root}")

# --- 执行批量处理 ---
if __name__ == "__main__":
    # 指向包含所有患者文件夹的总目录
    RAW_ROOT = r"data/raw/CN" 
    # 指向你想存放 NIfTI 文件的总目录
    NIFTI_ROOT = r"data/raw/CN/nii"
    
    batch_convert_dicom_to_nifti(RAW_ROOT, NIFTI_ROOT)