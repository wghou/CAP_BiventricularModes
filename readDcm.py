import numpy as np
import pydicom
from matplotlib import pyplot as plt

dcm = pydicom.read_file("data/IMG0010.dcm")

plt.figure(figsize=(12, 12))
plt.imshow(dcm.pixel_array, 'gray')
plt.show()