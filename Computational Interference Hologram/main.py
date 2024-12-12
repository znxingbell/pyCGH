import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 参数设置
image_size = 256  # 图像大小
wave_length = 0.632e-6  # 波长，单位m
pix = 0.00465 # CCD像素宽度
reference_angle = 0.5 # 参考光角度的sin值
reconstruct_angle = 0  # 重建角度的sin值
zoom = 200 # 缩放系数（调整此值以提高衬比度）


# 加载图像
image_path = 'pku.jpg'
image = Image.open(image_path).convert('L')
image = image.resize((image_size, image_size))
image_array = np.array(image)

# 展示初始图像
plt.imshow(image_array, cmap='gray')
plt.title('Custom Image')
plt.colorbar()
plt.show()

# 创建带有随机相位的物体场（模拟自然光漫反射）
random_phase = cp.random.rand(image_size, image_size)
object_field = cp.array(image_array) * cp.exp(1j * 2 * cp.pi * random_phase)

# 计算振幅和相位
hologram = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(object_field)))
hologram_amplitude = cp.abs(hologram)
hologram_amplitude = hologram_amplitude / cp.max(hologram_amplitude)
hologram_phase = cp.angle(hologram)

# 编码全息图
frequency_x =  reference_angle / wave_length
total_size = image_size * pix
x_range = cp.linspace(-total_size/2, total_size/2, image_size)
phase_factor = cp.cos(2 * cp.pi * frequency_x * x_range - hologram_phase)
encoded_hologram = (1 + hologram_amplitude * phase_factor) / 2

# 将编码后的全息图转换为CPU上的numpy数组以便显示
encoded_hologram_cpu = cp.asnumpy(encoded_hologram)

#展示编码后的全息图
plt.imshow(encoded_hologram_cpu, cmap='gray')
plt.title('Encoded Hologram')
plt.colorbar()
plt.show()

# 保存编码后的全息图
encoded_hologram_save = Image.fromarray((encoded_hologram_cpu * 255).astype(np.uint8))
encoded_hologram_save.save('encoded_hologram.png')

# 平行参考光重建图像
reconstruct_frequency = reconstruct_angle / wave_length
reference_field = cp.exp(1j * 2 * cp.pi * reconstruct_frequency * x_range)
reconstructed_field = cp.fft.fftshift(cp.fft.fft2(encoded_hologram * reference_field))
reconstructed_image = cp.abs(reconstructed_field)

# 处理重建图像以提高衬比度
re_max = cp.max(reconstructed_image)
re_min = cp.min(reconstructed_image)
cp.clip(reconstructed_image, re_min, re_max/zoom, out=reconstructed_image)
reconstructed_image = reconstructed_image / cp.max(reconstructed_image) * 255

# 将重建图像转换为CPU上的numpy数组以便显示
reconstructed_image_cpu = cp.asnumpy(reconstructed_image)

# 显示重建图像
plt.imshow(reconstructed_image_cpu, cmap='gray')
plt.title('Reconstructed Image')
plt.colorbar()
plt.show()

# 保存重建的图像
reconstructed_image_save = Image.fromarray((reconstructed_image_cpu / np.max(reconstructed_image_cpu) * 255).astype(np.uint8))
reconstructed_image_save.save('reconstructed_image.png')