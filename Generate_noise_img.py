from PIL import Image
import numpy as np

img = Image.open('U:/Summer intern project/CelebA/img_align_celeba/000001.jpg')

data = np.array(img)
print(data.shape)
noise = np.random.randn(218, 178, 3)
max_pixel = max(noise.reshape(-1, 1))
min_pixel = min(noise.reshape(-1, 1))
noise = (noise - min_pixel) / (max_pixel - min_pixel) * 256
noise = Image.fromarray(np.uint8(noise))

noise.save('noise.jpg')
