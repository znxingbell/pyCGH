import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 参数设置
image_size = 64  # 图像大小
unit_size = 16  # 光栅单元大小
wave_length = 0.632e-6  # 波长，单位m
pix = 0.00465 # CCD像素宽度
angle = 0 # 重建角度的sin值
zoom = 10 # 缩放系数（调整此值以提高衬比度）

# 加载图像
image_path = 'test.bmp'
image = Image.open(image_path).convert('L')
image = image.resize((image_size, image_size))
image_array = np.array(image)

# 展示初始图像
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.colorbar()
plt.show()

# 创建物体场并乘以随机相位（模拟自然光漫反射）
random_phase = cp.random.rand(image_size)
object_field = cp.array(image) * cp.exp(1j * 2 * cp.pi * random_phase)

# 计算振幅和相位
hologram = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(object_field)))
hologram_amplitude = cp.abs(hologram)
max_amplitude = cp.max(hologram_amplitude)
hologram_phase = cp.mod(cp.angle(hologram), 2 * cp.pi) / (2 * cp.pi)

# 初始化光栅
N = image_size * unit_size
grating = cp.zeros((N, N))

# 为每个像素分配光栅单元
for i in range(image_size):
    for j in range(image_size):
        # 计算矩形窗的大小和偏移位置
        amplitude = hologram_amplitude[i, j]
        phase = hologram_phase[i, j]
        window_width = amplitude * unit_size / max_amplitude
        window_height = unit_size / 2
        window_shift = cp.round(phase * unit_size)

        # 确定光栅单元的中心
        unit_end_x = (j + 1) * unit_size
        unit_start_x = j * unit_size
        center_x = j * unit_size + unit_size / 2
        center_y = i * unit_size + unit_size / 2

        # 确定矩形窗的边界
        start_x = center_x + window_shift - window_width / 2
        end_x = start_x + window_width
        start_y = center_y - window_height / 2
        end_y = start_y + window_height

        # 开窗
        if end_x < unit_end_x:
            grating[start_y:end_y, start_x:end_x] = 1
        else:
            grating[start_y:end_y, start_x:unit_end_x] = 1
            grating[start_y:end_y, unit_start_x:end_x - unit_size] = 1

# 将编码后的全息图转换为CPU上的numpy数组以便显示
encoded_hologram = cp.asnumpy(grating)

# 显示编码后的全息图
plt.imshow(encoded_hologram, cmap='gray')
plt.title('Encoded Hologram')
plt.colorbar()
plt.show()

# 保存编码后的全息图为图像
encoded_hologram_image = Image.fromarray((encoded_hologram * 255).astype(np.uint8))
encoded_hologram_image.save('encoded_hologram.png')

# 平行参考光重建图像
frequency_x = angle / wave_length
total_size = N * pix
x_range = cp.linspace(-total_size / 2, total_size / 2, N)
reference_field = cp.exp(1j * 2 * cp.pi * frequency_x * x_range)
reconstructed_field = cp.fft.fftshift(cp.fft.fft2(grating * reference_field))
reconstructed_image = cp.abs(reconstructed_field)

# 裁剪图像
cropped_image = reconstructed_image[N*3/8 : N*5/8, N*3/8 : N*5/8]

# 处理重建图像以提高衬比度
re_max = cp.max(cropped_image)
re_min = cp.min(cropped_image)
cp.clip(cropped_image, re_min, re_max / zoom, out=cropped_image)
cropped_image = cropped_image / cp.max(cropped_image) * 255

# 将重建图像转换为CPU上的numpy数组以便显示
reconstructed_image_cpu = cp.asnumpy(cropped_image)

# 显示重建图像
plt.imshow(reconstructed_image_cpu, cmap='gray')
plt.title('Reconstructed Image')
plt.colorbar()
plt.show()

# 保存重建的图像
reconstructed_image_save = Image.fromarray(reconstructed_image_cpu.astype(np.uint8))
reconstructed_image_save.save('reconstructed_image.png')