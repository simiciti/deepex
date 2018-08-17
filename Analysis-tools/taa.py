import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
sub1 = fig.add_subplot(121)
sub2 = fig.add_subplot(122)

x = np.linspace(0,10,200)
y = np.exp(x)
sub1.plot(x,y)

x = np.linspace(-10,10,200)
y = np.linspace(-10,10,200)
xx, yy = np.meshgrid(x,y)
z = np.sin(xx)+np.cos(yy)

img = sub2.imshow(z)
plt.colorbar(img, ax=sub2)
