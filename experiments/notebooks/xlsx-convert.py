import pandas as pd
import os

# 1. 设置文件夹路径
folder_path = '/home/wuzuoxu/Data/electric-power-dataset-test/raw/农网-数据pv_d-0225/辐射度数据' 
output_file = '/home/wuzuoxu/Data/electric-power-dataset-test/processed/节点2_1月-9月.xlsx'

# 2. 获取文件夹下所有 xlsx 文件
files = [f for f in os.listdir(folder_path) if f.endswith('.xls')]

all_data = []

for file in files:
    file_path = os.path.join(folder_path, file)
    # 读取数据，如果每个 xlsx 有多个 sheet，默认读取第一个
    df = pd.read_excel(file_path)
    # 可选：添加一列记录数据来源，方便溯源
    df['source_file'] = file
    all_data.append(df)

# 3. 使用 concat 合并
merged_df = pd.concat(all_data, ignore_index=True)
# 1. 确保时间列是真正的日期时间格式 (假设列名叫 '时间' 或 'w_start')
# 如果你的列名不同，请修改下面的 'w_start'
time_column = '北京时(UTC+8)' 

print(f"🕒 正在按 {time_column} 进行时间排序...")
merged_df[time_column] = pd.to_datetime(merged_df[time_column])

# 2. 执行升序排序
merged_df = merged_df.sort_values(by=time_column, ascending=True)

# 3. 重置索引（排序后行号会乱，重置后更整洁）
merged_df = merged_df.reset_index(drop=True)
# 4. 保存结果
merged_df.to_excel(output_file, index=False)
print(f"成功合并 {len(files)} 个文件至 {output_file}")