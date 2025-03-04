from tqdm import tqdm
import cupy as cp
import matplotlib.pyplot as plt
from PIL import Image

# 参数配置
wave_length = 532e-9  # 波长(m)
thetax, thetay = 0.2, 0.7  # 参考光角度(degree)
M, N = 512, 512  # 全息图分辨率
z_range = (1.1, 1.3)  # 物体深度范围(m)
pix = 24e-6  # 采样间隔(m)
num_layers = 100  # LUT分层数量


# LUT预生成函数
def generate_LUTs():
    z_values = cp.linspace(z_range[0], z_range[1], num_layers)
    # 生成坐标网格
    X = cp.arange(-N // 2, N // 2) * pix
    Y = cp.arange(-M // 2, M // 2) * pix
    x, y = cp.meshgrid(X, Y)
    LUTs = {}
    for z in z_values:
        # 生成球面波前相位
        phase = cp.pi / wave_length / z * (x ** 2 + y ** 2)
        # 添加随机初始相位
        LUTs[z.item()] = cp.exp(1j * (phase + cp.random.rand() * 2 * cp.pi))
    return LUTs, z_values


# 加载点云数据
coordinate = cp.loadtxt('bun000 - Cloud.txt').astype(cp.float32)
A = cp.ones(coordinate.shape[0], dtype=cp.float32)  # 振幅信息


# 数据预处理
def preprocess_points(points):
    """坐标预处理：缩放和平移"""
    rx, ry, rz = 0.1, 0.1, 0.1
    points[:, 0] *= rx
    points[:, 1] *= ry
    points[:, 2] = 1.2 - points[:, 2] * rz
    return points


coordinate = preprocess_points(coordinate)

# 生成LUT库
LUTs, z_samples = generate_LUTs()


# 全息图生成优化函数
def compute_hologram(points, LUTs, z_samples):
    """基于LUT法的全息图生成"""
    holo = cp.zeros((M, N), dtype=cp.complex64)

    # 将深度映射到最近的LUT层
    z_min, z_max = z_range
    z_indices = cp.clip(cp.searchsorted(z_samples, points[:, 2]), 0, num_layers - 1)

    # 批量处理点数据
    point_amount = coordinate.shape[0]
    for i in tqdm(range(point_amount), desc="全息图计算进度"):
        x0, y0 = coordinate[i, 0], coordinate[i, 1]
        base_LUT = LUTs[z_samples[z_indices[i]].item()]
        sx = int(x0 // pix) + M // 2
        sy = int(y0 // pix) + N // 2
        shifted_LUTs = cp.roll(cp.roll(base_LUT, sx, axis=1), sy, axis=0)
        holo += shifted_LUTs
    # 与参考光干涉
    X = cp.arange(-N // 2, N // 2) * pix
    Y = cp.arange(-M // 2, M // 2) * pix
    x, y = cp.meshgrid(X, Y)
    k = 2 * cp.pi / wave_length
    reference_phase = k * (x * cp.sin(cp.radians(thetax)) + y * cp.sin(cp.radians(thetay)))
    point_phase = cp.angle(holo)
    # 生成干涉图样
    interference = (1 + cp.cos(reference_phase - point_phase)) / 2
    return (interference - cp.min(interference)) / (cp.max(interference) - cp.min(interference))


# 计算并显示全息图
I = compute_hologram(coordinate, LUTs, z_samples)
plt.imshow(cp.asnumpy(I), cmap='gray')
plt.title('LUT Optimized Hologram')
plt.show()

# 重建图像
z = 1.2
k = 2 * cp.pi / wave_length
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