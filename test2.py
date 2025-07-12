import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

data = np.array([1, 1.1,1.2, 2, 2.2, 2.5, 3, 7, 9, 10])

# 合并数据集（保持原始顺序）
data = np.concatenate([data]).reshape(-1, 1)

print(data)


# 可视化密度分布
plt.figure(figsize=(10, 4))

# 原始数据点分布
# plt.subplot(1, 2, 1)
# plt.eventplot(data.ravel(), orientation='horizontal', colors='blue')
# plt.title('Sample Distribution')
# plt.xlim(-3, 12)

# 核密度估计
# plt.subplot(1, 2, 2)
# kde = KernelDensity(bandwidth=0.5).fit(data)
# x_vals = np.linspace(-3, 12, 1000).reshape(-1, 1)
# log_dens = kde.score_samples(x_vals)
# plt.fill_between(x_vals.ravel(), np.exp(log_dens), alpha=0.5)
# plt.title('Kernel Density Estimation')
# plt.tight_layout()
# plt.show()

# 打印原始数据
print("Raw data values:\n", np.sort(data.ravel()))


# DBSCAN示例
from sklearn.cluster import DBSCAN, HDBSCAN

db = HDBSCAN(min_cluster_size=2).fit(data)
print("Cluster labels:", db.labels_)  # 输出：[-1  0  0  0  0  1  1  1]


