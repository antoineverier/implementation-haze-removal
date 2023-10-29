from haze_functions import dark_channel, haze_remove
from haze_functions import atmosphere_light
from haze_functions import transmission

from skimage import io as skio
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hsv
import numpy as np



from cv2.ximgproc import guidedFilter
from guided_filter import guided_filter
#from cv2 import to_32F

im = skio.imread("haze_image/15.png")
im2 = skio.imread("haze_image/trees-1587301_1280.jpg")

darkchannel = dark_channel(im,15)

image_gray = np.mean(im, axis=2).astype(np.uint8)

lol = atmosphere_light(im, darkchannel, 0.001)

t = transmission(im, lol, 0.95, 15)

radius = 30
sigma = 0.0015   #0.015

im = im.astype(np.float32)
normI = (im - im.min()) / (im.max() - im.min())
darkchannel = darkchannel.astype(np.float32)
t = t.astype(np.float32)

#t = (t - t.min()) / (t.max() - t.min())

q = guided_filter(normI,t, radius, sigma)
q2 = guidedFilter(normI, t, radius, sigma)
im = im.astype(np.uint8)

q = (q - q.min()) / (q.max() - q.min())
q2 = (q2 - q2.min()) / (q2.max() - q2.min())
im = im.astype(np.uint8)
#t = t.astype(np.uint8)

final = haze_remove(im,lol,q,0.1)
final2 = haze_remove(im,lol,q2,0.1)

fig, axes = plt.subplots(1, 3, figsize=(25, 25))
axes[0].imshow(final)
axes[0].set_title('Image original')
axes[1].imshow(im, cmap='gray')
axes[1].set_title("transmission filtré à partir de l'image originale")
axes[2].imshow(q, cmap='gray')
axes[2].set_title('transmission')

#print(q)


plt.show()