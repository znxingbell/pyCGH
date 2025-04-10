from tqdm import tqdm
import cupy as cp
import matplotlib.pyplot as plt

# 参数配置
wave_length = 532e-9  # 波长(m)
thetax, thetay = 0.9, 0.3  # 参考光角度(degree)
M, N = 1024, 1024  # 全息图分辨率
z = 1.2
z_range = (z - 0.1, z + 0.1)  # 物体深度范围(m)
pix = 24e-6  # 采样间隔(m)
num_layers = 100  # LUT分层数量

# 加载点云数据
coordinate = cp.loadtxt('bun000 - Cloud.txt').astype(cp.float32)
#旋转点云
angle = cp.pi / 6
rotate_factor = cp.array([
    [cp.cos(angle).get(), 0, cp.sin(angle).get()],
    [0, 1, 0],
    [-cp.sin(angle).get(), 0, cp.cos(angle).get()]
], dtype=cp.float32)
coordinate = coordinate.astype(cp.float32) @ rotate_factor
def preprocess_points(points):
    rx, ry, rz = 0.1, 0.1, 0.1
    points[:, 0] *= rx
    points[:, 1] *= ry
    points[:, 2] = z - points[:, 2] * rz
    return points
coordinate = preprocess_points(coordinate)

# 生成LUT库
z_values = cp.linspace(z_range[0], z_range[1], num_layers)
X_LUT = cp.arange(-N // 2, N // 2) * pix
Y_LUT = cp.arange(-M // 2, M // 2) * pix
x_LUT, y_LUT = cp.meshgrid(X_LUT, Y_LUT)
LUTs = {}
for z_val in z_values:
    phase = cp.pi / wave_length / z_val * (x_LUT ** 2 + y_LUT ** 2)
    LUTs[z_val.item()] = cp.exp(1j * (phase + cp.random.rand() * 2 * cp.pi))
z_samples = z_values

# 计算扩展参数
max_x = cp.max(cp.abs(coordinate[:, 0]))
max_y = cp.max(cp.abs(coordinate[:, 1]))
padding_x = int(cp.ceil(max_x / pix))
padding_y = int(cp.ceil(max_y / pix))
M_new = M + padding_x * 2
N_new = N + padding_y * 2

# 预先计算深度索引
z_indices = cp.clip(cp.searchsorted(z_samples, coordinate[:, 2]), 0, num_layers - 1)

# 初始化扩展后的全息图
amplitude = cp.zeros((M_new, N_new), dtype=cp.complex64)

# 生成新的坐标网格和参考光相位
X_new = cp.arange(- (N_new // 2), N_new // 2) * pix
Y_new = cp.arange(- (M_new // 2), M_new // 2) * pix
x_new, y_new = cp.meshgrid(X_new, Y_new)
k = 2 * cp.pi / wave_length
sin_theta_x = cp.sin(cp.radians(thetax))
sin_theta_y = cp.sin(cp.radians(thetay))
reference_phase = k * (x_new * sin_theta_x + y_new * sin_theta_y)

# 复振幅分布计算
point_amount = coordinate.shape[0]
for i in tqdm(range(point_amount), desc="全息图计算进度"):
    x0, y0 = coordinate[i, 0], coordinate[i, 1]
    z_val = coordinate[i, 2]

    # 获取对应LUT相位
    z_sample = z_samples[z_indices[i]]
    base_LUT = LUTs[z_sample.item()]

    # 计算目标位置
    sx = int(x0 // pix) + (M // 2 + padding_x)
    sy = int(y0 // pix) + (N // 2 + padding_y)

    # 叠加到扩展后的全息图
    start_row = sx - (M // 2)
    end_row = sx + (M // 2)
    start_col = sy - (N // 2)
    end_col = sy + (N // 2)
    amplitude[start_row:end_row, start_col:end_col] += base_LUT

# 编码全息图
total_phase = cp.angle(amplitude)
interference = (1 + cp.cos(reference_phase - total_phase)) / 2
I = (interference - cp.min(interference)) / (cp.max(interference) - cp.min(interference))

# 显示全息图
plt.imshow(cp.asnumpy(I), cmap='gray')
plt.title('LUT Optimized Hologram')
plt.show()

# 重建图像
k = 2 * cp.pi / wave_length
L0 = wave_length * z / pix
I_1 = I
I = I_1 - cp.mean(I_1)

# 正确生成重建坐标网格
L_x = N_new * pix  # 水平方向总长度
L_y = M_new * pix  # 垂直方向总长度
x_recon = cp.linspace(-L_x / 2, L_x / 2, N_new)
y_recon = cp.linspace(-L_y / 2, L_y / 2, M_new)
X_recon, Y_recon = cp.meshgrid(x_recon, y_recon)
H = cp.exp(-1j * k / (2 * z) * (X_recon ** 2 + Y_recon ** 2))
U3 = I * H
U3 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(U3)))

# 重建时的坐标
x_recon = cp.linspace(-L0 / 2, L0 / 2, N_new)
y_recon = cp.linspace(-L0 / 2, L0 / 2, M_new)
X_recon, Y_recon = cp.meshgrid(x_recon, y_recon)
phase_recon = cp.exp(-1j * k * z) / (-1j * wave_length * z) * cp.exp(-1j * k / (2 * z) * (X_recon ** 2 + Y_recon ** 2))
U4 = U3 * phase_recon
reconstructed_image = cp.abs(U4)
re_max = cp.max(reconstructed_image)
re_min = cp.min(reconstructed_image)
cp.clip(reconstructed_image, re_min , re_max * 10 , out=reconstructed_image)
reconstructed_image = reconstructed_image / cp.max(reconstructed_image) * 255

# 显示重建图像
plt.imshow(cp.asnumpy(reconstructed_image), cmap='gray')
plt.title('Reconstructed Image')
plt.colorbar()
plt.show()