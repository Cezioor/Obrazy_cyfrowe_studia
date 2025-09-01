from skimage import color, data, restoration
img = color.rgb2gray(data.astronaut())
from scipy.signal import convolve2d
psf = np.ones((5, 5)) / 25
img = convolve2d(img, psf, 'same')
rng = np.random.default_rng()
img += 0.1 * img.std() * rng.standard_normal(img.shape)
deconvolved_img = restoration.wiener(img, psf, 0.1)