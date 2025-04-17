from tqdm import tqdm
import cupy as cp
import matplotlib.pyplot as plt

# parameters
wave_len = 532e-9 #wavelength
pix = 24e-6 #SLM unit interval
thx, thy = 0.2, 0.7  # reference light angle
z = 1.2 #average distance from SLM to CCD
M, N = 512, 512 #hologram resolution
mat = cp.array([ #rotation matrix
    [cp.cos(cp.pi / 6).get(), 0, cp.sin(cp.pi / 6).get()],
    [0, 1, 0],
    [-cp.sin(cp.pi / 6).get(), 0, cp.cos(cp.pi / 6).get()]
], dtype=cp.float32)
zoom = 0.1 #scale factor

#consts
vec = 2* cp.pi / wave_len #wave vector

#data pretreatment
data = cp.loadtxt('bun000 - Cloud.txt').astype(cp.float32)
data = data @ mat
data[:, 0] *= zoom
data[:, 1] *= zoom
data[:, 2] = z - data[:, 2] * zoom
data[:, 0] = (data[:, 0]/pix).astype(int)
data[:, 1] = (data[:, 1]/pix).astype(int)

#generate reference light
X = cp.arange(-N // 2, N // 2)
Y = cp.arange(-M // 2, M // 2)
x, y = cp.meshgrid(X, Y)
sin_thx = cp.sin(cp.radians(thx))
sin_thy = cp.sin(cp.radians(thy))
ref_ph = vec * (x * sin_thx + y * sin_thy) * pix

#generate s-lut
max_x = cp.max(cp.abs(data[:, 0]))
max_y = cp.max(cp.abs(data[:, 1]))
del_x = cp.arange(0, max_x + (M // 2)*pix, pix)
del_y = cp.arange(0, max_y + (N // 2)*pix, pix)
H = cp.exp(1j * vec * cp.sqrt(del_x**2 + z**2))
V = cp.exp(1j * vec * cp.sqrt(del_y**2 + z**2))

#generate hologram
amp = cp.zeros((M, N), dtype=cp.complex64)
point_amount = data.shape[0]
for i in tqdm(range(point_amount), desc="calculating"):
    x_idx = cp.abs(data[i, 0] - x).astype(int)
    y_idx = cp.abs(data[i, 1] - y).astype(int)
    amp += H[x_idx] * V[y_idx]
ph = cp.angle(amp)
inf = (1 + cp.cos(ref_ph - ph)) / 2
I = (inf - cp.min(inf)) / (cp.max(inf) - cp.min(inf))

# show hologram
plt.imshow(cp.asnumpy(I), cmap='gray')
plt.title('S-LUT Hologram')
plt.show()

# reconstruct image
L0 = wave_len * z / pix # 重建像平面宽度
I_1 = I
I = I_1 - cp.mean(I_1)
L = N * pix
x = cp.linspace(-L/2, L/2, N)
y = cp.linspace(-L/2, L/2, N)
X, Y = cp.meshgrid(x, y)
H = cp.exp(-1j * vec / (2 * z) * (cp.power(X, 2) + cp.power(Y, 2)))
U3 = I * H
U4 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(U3)))
x = cp.linspace(-L0/2, L0/2, N)
y = cp.linspace(-L0/2, L0/2, N)
X, Y = cp.meshgrid(x, y)
phase = cp.exp(-1j * vec * z) / (-1j * wave_len * z) * cp.exp(-1j * vec / 2 / z * (cp.power(X, 2) + cp.power(Y, 2)))
U4 = U4 * phase
reconstructed_image = cp.abs(U4)
reconstructed_image = reconstructed_image / cp.max(reconstructed_image) * 255

# 显示重建图像
plt.imshow(cp.asnumpy(reconstructed_image), cmap='gray')
plt.title('Reconstructed Image')
plt.colorbar()
plt.show()




