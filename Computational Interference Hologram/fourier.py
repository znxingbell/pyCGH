import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 参数设置
image_size = 256  # 图像大小
wave_length = 0.632e-6  # 波长，单位m
pix = 4.65e-6 # CCD像素宽度，单位m
N = 1024  # 采样率
distance = 1  # 传播距离，单位m
L = N * pix         # CCD宽度
zoom = 1 # 缩放系数（调整此值以提高衬比度）

# 创建空间滤波器


# 加载图像
image_path = 'pku.jpg'
image = Image.open(image_path).convert('L')
image = image.resize((image_size, image_size))
image_array = np.array(image)

# 匹配CCD平面的大小
padded_array = cp.zeros((N, N))
start = (N - image_size) // 2
padded_array[start:start+image_size, start+128:start+image_size+128] = cp.array(image_array)

# 展示原始图像
image_array = cp.asnumpy(padded_array)
plt.imshow(image_array, cmap='gray')
plt.title('Origin Image')
plt.colorbar()
plt.show()

# 创建带有随机相位的物体场（模拟自然光漫反射）
random_phase = cp.random.rand(N, N)
object_field = padded_array * cp.exp(1j * 2 * cp.pi * random_phase)

# 计算传播到CCD平面的物体场
fft_object_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(object_field)))
mask = cp.zeros((N, N))
center = N // 2
mask[:, :center] = 1
fre = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(fft_object_field)))
fre = fre * mask
fft_object_field = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(fre)))

# 展示振幅分布
image_array = cp.asnumpy(cp.abs(fft_object_field))
plt.imshow(image_array, cmap='gray')
plt.title('amplitude Image')
plt.colorbar()
plt.show()

# 参考光
k = 2 * cp.pi / wave_length
x = cp.linspace(-L/2, L/2 - L/N, N)
y = cp.linspace(-L/2, L/2 - L/N, N)
X, Y = cp.meshgrid(x, y)
reference_light = cp.max(cp.abs(fft_object_field)) * cp.exp(1j * k * (X * 0 + Y * 0 ))

# 编码全息图
object_amplitude = cp.abs(fft_object_field) / cp.max(cp.abs(fft_object_field))
object_phase = cp.angle(fft_object_field)
reference_phase = cp.angle(reference_light)
encoded_hologram = (1 + object_amplitude * cp.cos(reference_phase - object_phase)) / 2 * 255

# 将编码后的全息图转换为CPU上的numpy数组以便显示
encoded_hologram_cpu = cp.asnumpy(encoded_hologram)

#展示编码后的全息图
plt.imshow(encoded_hologram_cpu, cmap='gray')
plt.title('Encoded Hologram')
plt.colorbar()
plt.show()

# 保存编码后的全息图
encoded_hologram_save = Image.fromarray((encoded_hologram_cpu).astype(np.uint8))
encoded_hologram_save.save('encoded_hologram.png')

#空间滤波
fre = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(encoded_hologram)))
mask = cp.zeros((N, N))
center = N // 2
mask[:, :center] = 1
fre = fre * mask
reconstructed_image = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(fre)))

# 重建图像
reconstructed_image = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(reconstructed_image)))
reconstructed_image = cp.abs(reconstructed_image)

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