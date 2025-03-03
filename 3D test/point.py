from tqdm import tqdm
import cupy as cp
import matplotlib.pyplot as plt
from PIL import Image

# 参数设置
wave_length = 532e-9  # 波长 单位m
thetax, thetay = 0.2, 0.7  # 参考光角度
M, N = 512, 512
z = 1.2  # 全息图与物体的距离 单位m
pix = 24e-6  # 全息图采样间隔 单位m

# 加载数据（点云模型）==============================================
# 使用cupy直接加载数据（需保证原始数据是数值格式）
coordinate = cp.loadtxt('bun000 - Cloud.txt')

# 优化旋转矩阵计算（完全在GPU上执行）
angle = cp.pi / 6
rotate_factor = cp.array([
    [cp.cos(angle).get(), 0, cp.sin(angle).get()],
    [0, 1, 0],
    [-cp.sin(angle).get(), 0, cp.cos(angle).get()]
], dtype=cp.float32)  # 使用float32提升计算速度
coordinate = coordinate.astype(cp.float32) @ rotate_factor  # 保持数据类型一致

# 缩放模型
rx, ry, rz = 0.1, 0.1, 0.1 # 缩放因子
coordinate[:, 0] *= rx
coordinate[:, 1] *= ry
coordinate[:, 2] = z - coordinate[:, 2] * rz

# 生成坐标网格=====================================================
X = cp.arange(-N // 2, N // 2, dtype=cp.float32) * pix  # 使用float32
Y = cp.arange(-M // 2, M // 2, dtype=cp.float32) * pix
x, y = cp.meshgrid(X, Y, sparse=True)  # 使用稀疏网格减少内存占用

# 生成参考光=======================================================
k = 2 * cp.pi / wave_length
reference_phase = k * (x * cp.sin(cp.radians(thetax)) + y * cp.sin(cp.radians(thetay)))

# 计算全息图=======================================================
h = cp.zeros((M, N), dtype=cp.float32)
point_amount = coordinate.shape[0]
factor1 = cp.pi / wave_length / z
factor2 = x**2 + y**2
for i in tqdm(range(point_amount), desc="全息图计算进度"):
    x0, y0, z0 = coordinate[i, 0], coordinate[i, 1], coordinate[i, 2]
    p0 = cp.random.rand() * 2 * cp.pi
    point_phase = p0 + factor1 * (factor2 + (x0**2 + y0**2) - 2 * (x0 * x + y0 * y))
    h += (1 + cp.cos(reference_phase - point_phase)) / 2
minh, maxh = cp.min(h), cp.max(h)
I = (h - minh) / (maxh - minh)

# 显示全息图
plt.imshow(cp.asnumpy(I), cmap='gray')
plt.title('Encoded Hologram')
plt.colorbar()
plt.show()

# 保存全息图
reconstructed_image_save = Image.fromarray((cp.asnumpy(I) * 255).astype(cp.uint8))
reconstructed_image_save.save('hologram_image.png')

# 重建图像
L0 = wave_length * z / pix # 重建像平面宽度
I_1 = cp.array(Image.open('hologram_image.png').convert('L'), dtype=cp.float32)
I = I_1 - cp.mean(I_1)
L = N * pix
x = cp.linspace(-L/2, L/2, N)
y = cp.linspace(-L/2, L/2, N)
X, Y = cp.meshgrid(x, y)
H = cp.exp(-1j * k / (2 * z) * (cp.power(X, 2) + cp.power(Y, 2)))
U3 = I * H
U4 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(U3)))
x = cp.linspace(-L0/2, L0/2, N)
y = cp.linspace(-L0/2, L0/2, N)
X, Y = cp.meshgrid(x, y)
phase = cp.exp(-1j * k * z) / (-1j * wave_length * z) * cp.exp(-1j * k / 2 / z * (cp.power(X, 2) + cp.power(Y, 2)))
U4 = U4 * phase
reconstructed_image = cp.abs(U4)
reconstructed_image = reconstructed_image / cp.max(reconstructed_image) * 255

# 显示重建图像
plt.imshow(cp.asnumpy(reconstructed_image), cmap='gray')
plt.title('Reconstructed Image')
plt.colorbar()
plt.show()

# 保存重建的图像
reconstructed_image_save = Image.fromarray((cp.asnumpy(reconstructed_image)).astype(cp.uint8))
reconstructed_image_save.save('reconstructed_image.png')