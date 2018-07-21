import io, os, shutil, glob
import random

experiments = [fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and '_' in fn]
for exp in experiments:
    frames = glob.glob('*.tif')
    used = [-1]
    selected = -1
    for i in range(100):
        while selected in used:
            selected = int(random.random() * len(frames))
        shutil.copy(frames[selected], '..\\trainingset\\' + frames[selected])
        used.append(selected)
        
        
