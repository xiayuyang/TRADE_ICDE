import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 假设的驾驶时间数据（单位：分钟）
data_c1_to_c2 = np.random.normal(20, 5, 1000)  # 假设c1到c2平均20分钟，标准差5
data_c1_to_c3 = np.random.normal(30, 10, 1000)  # 假设c1到c3平均30分钟，标准差10

# 合并数据以计算总概率密度
all_data = np.concatenate([data_c1_to_c2, data_c1_to_c3])

# 使用Gaussian Kernel Density Estimation计算概率密度
kde_c1_to_c2 = gaussian_kde(data_c1_to_c2)
kde_c1_to_c3 = gaussian_kde(data_c1_to_c3)
kde_all = gaussian_kde(all_data)

# 为概率密度函数定义一个时间范围
time_range = np.linspace(0, 60, 500)

# 计算每个时间点的概率密度
pdf_c1_to_c2 = kde_c1_to_c2(time_range) / kde_all(time_range)
pdf_c1_to_c3 = kde_c1_to_c3(time_range) / kde_all(time_range)

# 绘制概率密度曲线
plt.figure(figsize=(10, 6))
plt.plot(time_range, pdf_c1_to_c2, label='Probability c1 to c2', linewidth=2)
plt.plot(time_range, pdf_c1_to_c3, label='Probability c1 to c3', linewidth=2)

plt.xlabel('Time (minutes)')
plt.ylabel('Probability')
plt.title('Probability of Arrival Time from c1 to c2 and c1 to c3')
plt.legend()

# plt.show()

plt.savefig('distributions.png')
