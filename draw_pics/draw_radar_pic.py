# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

# 使用ggplot的绘图风格
plt.style.use('ggplot')
# 构造数据
values = [0.53, 0.50, 0.43, 0.51]
values2 = [0.7647, 0.8125, 0.77, 0.788]
values3 = [0.35, 0.46, 0.64, 0.40]
values4 = [0.67, 0.31, 0.78, 0.42]
values5 = [0.38, 0.32, 0.54, 0.35]
values6 = [0.50, 0.37, 0.62, 0.43]
feature = ['Precision', 'Recall', 'Accuracy', 'f1-score']

N = len(values)
# 设置雷达图的角度，用于平分切开一个圆面
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
# 为了使雷达图一圈封闭起来，需要下面的步骤
values = np.concatenate((values, [values[0]]))
values2 = np.concatenate((values2, [values2[0]]))
values3 = np.concatenate((values3, [values3[0]]))
values4 = np.concatenate((values4, [values4[0]]))
values5 = np.concatenate((values5, [values5[0]]))
values6 = np.concatenate((values6, [values6[0]]))
angles = np.concatenate((angles, [angles[0]]))

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
# 绘制折线图
ax.plot(angles, values, 'o-', linewidth=2, label='西班牙，Google Trends，1个月')
# 填充颜色
ax.fill(angles, values, alpha=0.25)
# 绘制第二条折线图
ax.plot(angles, values2, 'o-', linewidth=2, label='西班牙，Google Trends，1个月，预测走势框架')
ax.fill(angles, values2, alpha=0.25)

# ax.plot(angles, values3, 'o-', linewidth=2, label='深度学习，不加纽约时报，1个月')
# ax.fill(angles, values3, alpha=0.25)
#
# ax.plot(angles, values4, 'o-', linewidth=2, label='深度学习，加入纽约时报，1个月')
# ax.fill(angles, values4, alpha=0.25)
#
# ax.plot(angles, values5, 'o-', linewidth=2, label='XGB,不加纽约时报，3个月')
# ax.fill(angles, values5, alpha=0.25)
#
# ax.plot(angles, values6, 'o-', linewidth=2, label='XGB，加入纽约时报，3个月')
# ax.fill(angles, values6, alpha=0.25)

# 添加每个特征的标签
ax.set_thetagrids(angles * 180 / np.pi, feature)
# 设置雷达图的范围
ax.set_ylim(0, 0.9)
# 添加标题
plt.title('2004年至今加入Google Trends对西班牙进行预测的结果')

# 添加网格线
ax.grid(True)
# 设置图例
plt.legend(loc='lower left', bbox_to_anchor=(0.6, 0), ncol=1, fancybox=True, shadow=True)
# 显示图形
plt.savefig("3.png")
plt.show()
