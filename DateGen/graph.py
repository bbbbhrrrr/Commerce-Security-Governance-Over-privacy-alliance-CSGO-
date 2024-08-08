import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 定义表达式
def expression(x, y, z, a, b, c):
    return a * x + b ** y + c ** z

# 设置初始参数值
a_init = 1
b_init = 2
c_init = 3

# 定义y, z的范围，x可以固定
y = np.linspace(1, 10, 100)
z = np.linspace(1, 10, 100)
x_fixed = 1  # 固定x值

# 创建网格
Y, Z = np.meshgrid(y, z)
T = expression(x_fixed, Y, Z, a_init, b_init, c_init)

# 创建图形和轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制初始图像，设置y和z为横轴，t为纵轴
plot = ax.plot_surface(Y, Z, T, cmap='viridis')

# 设置轴标签
ax.set_xlabel('y')
ax.set_ylabel('z')
ax.set_zlabel('t')

# 添加滑动条来控制a, b, c的值
ax_a = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_c = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_a = Slider(ax_a, 'a', -10.0, 10.0, valinit=a_init)
slider_b = Slider(ax_b, 'b', -10.0, 10.0, valinit=b_init)
slider_c = Slider(ax_c, 'c', -10.0, 10.0, valinit=c_init)

# 更新函数
def update(val):
    a = slider_a.val
    b = slider_b.val
    c = slider_c.val
    T = expression(x_fixed, Y, Z, a, b, c)
    ax.clear()
    ax.plot_surface(Y, Z, T, cmap='viridis')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('t')
    plt.draw()

# 为滑动条绑定更新函数
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_c.on_changed(update)

plt.show()
