import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

a=[]
b=[]
time_a = []
time_b = []
#读入数据
df = pd.read_csv('1.csv')
for i in range(len(df['point_victor'])):
    speed = df['speed_mph'][i]
    if speed < 20:  # 如果速度小于20，跳过这个值
        continue
    time = datetime.strptime(df['elapsed_time'][i], "%H:%M:%S")  # 将字符串转换为datetime对象
    total_seconds = time.hour * 3600 + time.minute * 60 + time.second  # 将时间转换为总秒数
    if(df['point_victor'][i]==1):
        a.append(speed)
        time_a.append(total_seconds)
    else:
        b.append(speed)
        time_b.append(total_seconds)

plt.figure(figsize=(10, 6))

# 绘制动量随时间的变化曲线
plt.scatter(time_a, a, label='Carlos Alcaraz')
plt.scatter(time_b, b, label='Nicolas Jarry')

# 添加标题和标签
plt.title('Ball Speed Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Ball Speed')

# 添加图例
plt.legend()

# 显示图表
plt.show()