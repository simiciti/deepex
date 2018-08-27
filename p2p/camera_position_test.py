import imageio
import sys
import numpy as np




if __name__ == '__main__':
    im = imageio.imread(sys.argv[1])
    if np.average(im[230:240, 175]) > 220:
          print('activated')
          
     elif np.average(im[230:240, 505]) > 220:
          print('moved')
     else:
          print('null')
