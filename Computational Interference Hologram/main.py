#这个还没写好，别看（
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 参数设置
image_size = 256  # 图像大小

# 加载自定义图像
image_path = './pku.jpg'
image = Image.open(image_path).convert('L')
image = image.resize((image_size, image_size))
image_array = np.array(image)

# 显示加载的图像
plt.imshow(image_array, cmap='gray')
plt.title('Custom Image')
plt.colorbar()
plt.show()

# 创建带有随机相位的物体场（模拟自然光漫反射）
random_phase = cp.random.rand(image_size, image_size)
object_field = cp.array(image_array) * cp.exp(1j * 2 * cp.pi * random_phase)

# 计算傅里叶变换以获得全息图
hologram = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(object_field)))
hologram_amplitude = cp.abs(hologram)
hologram_phase = cp.angle(hologram)

# 保存全息图的振幅和相位
hologram_amplitude_cpu = cp.asnumpy(hologram_amplitude)
hologram_phase_cpu = cp.asnumpy(hologram_phase)

# 显示全息图振幅
plt.imshow(hologram_amplitude_cpu, cmap='gray')
plt.title('Hologram Amplitude')
plt.colorbar()
plt.show()

# 保存全息图振幅为图像
hologram_amplitude_image = Image.fromarray((hologram_amplitude_cpu / np.max(hologram_amplitude_cpu) * 255).astype(np.uint8))
hologram_amplitude_image.save('./hologram_amplitude.png')

# 使用逆傅里叶变换重建图像
reconstructed_field = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(hologram)))
reconstructed_image = cp.abs(reconstructed_field)

# 将重建图像转换为CPU上的numpy数组以便显示
reconstructed_image_cpu = cp.asnumpy(reconstructed_image)

# 显示重建图像
plt.imshow(reconstructed_image_cpu, cmap='gray')
plt.title('Reconstructed Image')
plt.colorbar()
plt.show()

# 保存重建的图像
reconstructed_image_save = Image.fromarray((reconstructed_image_cpu / np.max(reconstructed_image_cpu) * 255).astype(np.uint8))
reconstructed_image_save.save('./reconstructed_image.png')