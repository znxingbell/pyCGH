import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 参数设置
image_size = 256  # 图像大小
wavelength = 0.532e-3  # 波长(mm)
k = 2 * cp.pi / wavelength  # 波数
pix = 0.0064  # SLM像素宽度(mm)
N = 512  # 相息图取样数
L = N * pix  # SLM宽度(mm)
m0 = 0.5  # 图像在重建周期中的显示比例
z0 = 1200  # 衍射距离(mm)
L0 = wavelength * z0 / pix  # 重建像平面宽度(mm)

# 加载图像
image_path = 'BIT.jpg'
image = Image.open(image_path).convert('L')
image = image.resize((image_size, image_size))
image_array = cp.array(image)

# 裁剪图像
M1, N1 = image_array.shape
X = cp.zeros((N, N))
X[N//2-M1//2:N//2+M1//2, N//2-N1//2:N//2+N1//2] = image_array

# 显示原始图像
plt.figure()
plt.imshow(cp.asnumpy(X), cmap='gray')
plt.title('Original Image')
plt.colorbar()
plt.show()

# 初始场随机相位
Y = X.astype(cp.float64)
a = cp.ones((N, N))
b = cp.random.rand(N, N) * 2 * cp.pi
U0 = Y * cp.exp(1j * b)
X0 = cp.abs(U0)

# 迭代参数
np_iter = int(input('Number of iterations: '))

for p in range(np_iter + 1):
    # 菲涅耳衍射 (S-FFT)
    x_coords = cp.linspace(-L0 / 2, L0 / 2, N)
    y_coords = cp.linspace(-L0 / 2, L0 / 2, N)
    xx, yy = cp.meshgrid(x_coords, y_coords)
    Fresnel = cp.exp(1j * k / (2 * z0) * (xx**2 + yy**2))
    f2 = U0 * Fresnel
    Uf = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(f2)))
    x_coords = cp.linspace(-L / 2, L / 2, N)
    y_coords = cp.linspace(-L / 2, L / 2, N)
    xx, yy = cp.meshgrid(x_coords, y_coords)
    phase = cp.exp(1j * k * z0) / (1j * wavelength * z0) * cp.exp(1j * k / (2 * z0) * (cp.power(xx, 2) + cp.power(yy, 2)))
    Uf = Uf * phase

    # 显示SLM平面的物光振幅分布
    plt.figure()
    plt.imshow(cp.asnumpy(cp.abs(Uf)), cmap='gray')
    plt.title('Amplitude Distribution at SLM Plane')
    plt.colorbar()
    plt.show()

    # 生成相息图
    Phase = cp.angle(Uf) + cp.pi
    Ih = (Phase / (2 * cp.pi) * 255)
    plt.figure()
    plt.imshow(cp.asnumpy(Phase).astype(cp.uint8), cmap='gray')
    plt.title('Hologram')
    plt.colorbar()
    plt.show()

    # 逆菲涅耳衍射 (S-IFFT)
    U0 = cp.exp(1j * cp.angle(Uf))
    x_coords = cp.linspace(-L / 2, L / 2, N)
    y_coords = cp.linspace(-L / 2, L / 2, N)
    xx, yy = cp.meshgrid(x_coords, y_coords)
    Fresnel = cp.exp(-1j * k / (2 * z0) * (cp.power(xx, 2) + cp.power(yy, 2)))
    f2 = U0 * Fresnel
    Uf = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(f2)))
    x_coords = cp.linspace(-L0 / 2, L0 / 2, N)
    y_coords = cp.linspace(-L0 / 2, L0 / 2, N)
    xx, yy = cp.meshgrid(x_coords, y_coords)
    phase = cp.exp(-1j * k * z0) / (-1j * wavelength * z0) * cp.exp(-1j * k / (2 * z0) * (xx**2 + yy**2))
    Uf = Uf * phase

    # 显示逆运算重建的物平面振幅分布
    plt.figure()
    plt.imshow(cp.asnumpy(cp.abs(Uf)), cmap='gray')
    plt.title('Reconstructed Amplitude Distribution')
    plt.colorbar()
    plt.show()

    # 更新U0以进行下一次迭代
    Phase = cp.angle(Uf)
    U0 = X0 * cp.exp(1j * Phase)

# 显示最终相息图
plt.figure()
plt.imshow(cp.asnumpy(Ih), cmap='gray')
plt.title('Hologram')
plt.colorbar()
plt.show()

# 保存相息图
Ih = cp.asnumpy(Ih)
hologram = Ih / cp.max(Ih) * 255
hologram_image_save = Image.fromarray(hologram.astype(np.uint8))
hologram_image_save.save('hologram_image.png')

#保存重建图像
Uf = cp.abs(Uf)
Uf = cp.asnumpy(Uf)
hologram = Uf / cp.max(Uf) * 255
hologram_image_save = Image.fromarray(hologram.astype(np.uint8))
hologram_image_save.save('reconstructed_image.png')
