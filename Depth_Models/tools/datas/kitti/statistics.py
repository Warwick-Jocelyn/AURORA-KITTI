import pandas as pd

# 你的CSV路径
csv_path = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/test.csv"

# 读取
df = pd.read_csv(csv_path)

# 清理字符串字段（去掉前后空格，统一缺失）
for col in ["weather", "Time_of_Day"]:
    df[col] = df[col].astype(str).str.strip()
df["weather"] = df["weather"].replace({"None": pd.NA, "nan": pd.NA, "": pd.NA})
df["Time_of_Day"] = df["Time_of_Day"].replace({"None": pd.NA, "nan": pd.NA, "": pd.NA})

# 只统计我们关心的四种天气（Clear/Fog/Rain/Snow）
valid_weather = {"Clear", "Fog", "Rain", "Snow"}
df = df[df["weather"].isin(valid_weather) & df["Time_of_Day"].isin(["Day", "Night"])].copy()

# 生成组合标签：Day-Clear 之类
df["bucket"] = df["Time_of_Day"] + "-" + df["weather"]

# 按要求固定顺序输出（没有的类别补0）
order = [
    "Day-Clear", "Day-Fog", "Day-Rain", "Day-Snow",
    "Night-Clear", "Night-Fog", "Night-Rain", "Night-Snow"
]
counts = df["bucket"].value_counts().reindex(order, fill_value=0)

print(counts)
print("\nTotal:", int(counts.sum()))
