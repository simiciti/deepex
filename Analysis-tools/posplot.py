import os.path
import sys
import io

import matplotlib.pyplot as plt
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1 import make_axes_locatable

subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(subfolder + "/Generating_a_Training_Set")

import numpy as np

from myconfig import bodyparts, center_of_mass
from myconfig_analysis import videofolder


import seaborn as sns

def distance(x1, y1, x2, y2):
    '''
    Distance betweeen 2 points.
    '''
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def tridis(triangle,centx=0,centy=0):
    for point in triangle:
        print('Point: ',point, distance(centx,centy, point[0], point[1]))
    
def display_triangle(tri):
    '''
    Used for testing. In actual version, triangle should be superimposed
    over image 
    '''
    x,y = gen_display_triangle(tri)
    fig, ax = plt.subplots()
    line, = ax.plot(x,y)
    plt.xlim(-40,300)
    plt.ylim(-40,300)
    plt.show()

def gen_display_triangle(triangle):
    Path = mpath.Path

    path_data = []

    path_data.append((Path.MOVETO, (triangle[0][0], triangle[0][1])))
    path_data.append((Path.LINETO, (triangle[1][0], triangle[1][1])))
    path_data.append((Path.LINETO, (triangle[2][0], triangle[2][1])))
    path_data.append((Path.CLOSEPOLY, (triangle[0][0], triangle[0][1])))

    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)

    x,y = zip(*path.vertices)
    return x, y

    
def gen_triangle(x, y, bearing, dist=77):
    '''
    Generates an equilateral triangle centered around the given (x, y) point,
    oriented at the given bearing (in radians)
    '''

    sqrt3 = np.sqrt(3)

    front = np.asarray([dist / sqrt3, 0])
    left = np.asarray([-dist / (2 * sqrt3), dist / 2])
    right = np.asarray([-dist / (2 * sqrt3), -dist / 2])

    points = np.asarray([front, left, right])
    #inverting owing to inverted image y-axis, maybe not necessary
    transform = np.asarray([[np.cos(bearing), np.sin(bearing)],
                            [-np.sin(bearing), np.cos(bearing)]])
    return np.matmul(points, transform) + np.asarray([x,y] * 3).reshape((3,2))
    
def load_triangles(posfile,thetafile):
    '''
    Loads files, for each entry (each frame), generates triangle
    with appropriate coordinate points and orientation
    '''
    # Check that the files have the same length
    # Considering what comes next, really just a courtesy
    if len(posfile) != len(thetafile):
        print('Position and Orientation file lengths do not match!')
        
    frames = min(len(posfile), len(thetafile))
    triangles = []
    with open(posfile, 'r') as p:
        with open(thetafile, 'r') as t:
            pos = p.readlines()
            thetas = t.readlines()
    for i in range(frames):
        x, y = float(pos[i].split(',')[0]), float(pos[i].split(',')[1])
        theta = float(thetas[i])
        triangles.append(gen_triangle(x, y, theta))
    return triangles
        
def display_triangles(triangles, movie):
    
    for i in range(len(triangles)):
        x_lst = []
        y_lst = []
        for j in range(len(triangles[i])):
            #should be 3
            x_lst.append(triangles[i][j][0])
            y_lst.append(triangles[i][j][1])
        #overlay triangle upon actual for comparison
        # save to file
    
def read_data(folder):
    '''
        Read Position and bearing data from folder
    '''
    with open(folder[:-8] +'_position.csv', 'r') as f:
        position = f.readlines()

    with open(folder[:-8] + '_bearing.csv', 'r') as f:
        bearing = [float(b) for b in f.readlines()]

    position =[(float(p.split(',')[0]),float(p.split(',')[1])) for p in position]

    return np.asarray(position), np.asarray(bearing)
    
    
def heatmap(pos, folder, imname='', save=True, **kw):
    '''
    Converts trajectory data to pixel-based heatmap
    agnostic of whether stimulus was present or mouse was activated.
    '''
    #plt.title('Location Heatmap')
    # Find the limits in the x and y directions; These will be the boundaries
    # of the frame 
    [xlim, ylim] = np.nanmax(pos, axis=0)
    xlim, ylim = int(xlim) + 20, int(ylim) + 20

    # Initialize cells
    cells = np.zeros((xlim,ylim))


    #placeholder value - set cell to 1 if the mouse was at it
    increment = 1
    for i in range(len(pos)):
        cells[int(pos[i][0])][int(pos[i][1])] = increment
    
    ax = kw.get('ax')
    #transpose because it inverts the display (but not the actual graph)
    #plt.title('loc')
    im = ax.imshow(cells.T, cmap='YlGn')
    
    plt.xlabel('pixels') 
    plt.ylabel('pixels')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)


    #plt.xlim(0, xlim)
    #plt.ylim(0, ylim)
    plt.ylabel('Number of Occurrencs')
    
    if save:
        plt.savefig(folder + '/' + imname + '_heatmap.png')

def segmented(pos, function, folder, seg, name='', labels=None, **kw):
    '''
    Provides an easy interface for segmenting the position data and
    applying any properly parameterized function to the data.
    '''
    if kw.get('aggregate'):
        epochs = {}
        for label in labels:
            epochs[label] = []
            
        for i in range(1, len(seg) + 1):
            cur = seg[i] if i < len(seg) else None
            lo = seg[i - 1]

            if epochs[labels[i - 1]] != []:
                epochs[labels[i - 1]] = np.concatenate((epochs[labels[i - 1]],pos[lo: cur]))
            else:
                epochs[labels[i - 1]] = pos[lo:cur]
            
        for lbl in epochs.keys():
            if kw.get('verbose'):
                print(lbl, epochs[lbl].shape())
            function(epochs[lbl], folder, name + ' ' + lbl , kw)
    else:
         for i in range(1, len(seg) + 1):
            cur = seg[i] if i < len(seg) else None
            lo = seg[i - 1]

            epoch = pos[lo: cur]
            lbl = labels[i - 1] if labels else ''
            function(epoch, folder, str(lo) + '-' + str(cur) + ' ' + lbl, kw)
    

'''    
def data_to_heatmap_seg(pos, folder, seg, labels=None):
    \'''
    Creates heatmaps of mouse position segmented by epoch boundaries
    contained in list seg. Saves heatmaps to folders. labels is an optional
    list of labels for the epochs. 
    \'''

    
    for i in range(1, len(seg)):
        cur = seg[i]
        lo = seg[(i - 1) * (( i - 1) > 0)] * ((i - 1) > 0)

        epoch = pos[lo: cur]
        lbl = labels[i - 1] if labels else ''
        
        data_to_heatmap(epoch, folder, lo + '-' + cur + ' ' + lbl)
  '''      

    
def plot_trajectory(pos, folder, name='', save=True, **kw):
    '''
    Be a right-sider. Connect the cases.
    opacity: 0.2
    
    '''
    #plt.title('Trajectory')
    #[xlim, ylim] = np.nanmax(pos, axis=0)
    #xlim, ylim = int(xlim) + 20, int(ylim) + 20

    opacity = 0.2
    
    if kw['index']:
        index = kw['index']
        #plt.subplot(131)
        ax = plt.subplot(index[0],index[1],index[2])
        plotted = plt.plot(pos[:,0],pos[:,1], alpha=opacity, linewidth=0.25)
        ax.set_aspect(1)
        
        
    else:
        plotted = plt.plot(pos[:,0],pos[:,1], alpha=opacity)
    #plt.title('Trajectory')
    #plt.xlim(0, xlim)
    #plt.ylim(0, ylim)
    plt.xlabel('pixels')
    plt.ylabel('pixels')
        
    if save:
        plt.savefig(folder + '/' + name + '_traj.png')
        
#def plot_trajectory_seg():
#    return None 
    
def kde(pos, folder, name='', save=True, **kw):
    '''
    Plots a kernel density estimate for the position data
    '''
    #plt.title('Kernel Density Estimate')
    #[xlim, ylim] = np.nanmax(pos, axis=0)
    #xlim, ylim = int(xlim) + 20, int(ylim) + 20
    
    #create cubehelix colormap
    cmap = sns.cubehelix_palette(start=3, light=1, as_cmap=True)

    #actual plot

    plt.xlabel('pixels')
    plt.ylabel('pixels')
    
    s = sns.kdeplot(pos[:,0], pos[:,1], cmap=cmap, shade=True,ax=kw.get('ax'))
    
    #s = sns.kdeplot(pos, cmap=cmap, shade=True,ax=kw.get('ax'))

    
    #plt.xlim(0, xlim)
    #plt.ylim(0, ylim)
    
    if save:
        plt.savefig(folder + '/' + name + '_kde.png')
    
def multiplot(pos, folder, name, save=True, display=False):
    '''
    Plots multiple representations of position data for a single experiment
    '''

    [xlim, ylim] = np.nanmax(pos, axis=0)
    xlim, ylim = int(xlim) + 20, int(ylim) + 20

    functions = [heatmap, plot_trajectory, kde]
    titles = ['Kernel Density Estimate','','Trajectory']
    
    f, axes = plt.subplots(1,len(functions), figsize=(19,6), sharex=True, sharey=True)

    #create grid of subplots
    
    for ax, s in zip(axes.flat, np.linspace(0,len(functions) - 1,len(functions))):
        plt.title(titles[int(s)])
        functions[int(s)](pos, folder, name, save=False, ax=ax,index=(1,len(functions),int(s) + 1))
        ax.set(xlim=(0,xlim), ylim=(0,ylim))
        ax.set_aspect(1)
        

        
        
    
    f.tight_layout(pad=1.08)
    plt.suptitle(name)
    
    if save:
        if folder:
            plt.savefig(folder + '/' + name + '_a.png')
        else:
            plt.savefig(name + '_a.png')

    if display:
        plt.show()

    

        



#def kde_seg():
#    return None 



#depreciated
def bearing_sunspot(bearing, title):
    '''
    convert to int
    count frequency of each orientation
    create 1deg arc at base radius
    create 1deg arc scaled radius
    add labels at k* n.pi / 4
    retrospect: not helpful 
    '''

    buckets = np.zeros((360,1))
    for theta in bearing:
        buckets[int(360 * theta / (2*np.pi))] += 1 
    
    return 'sf'

               
               
               

    
'''
# This is correct, right? Hisashiburi, ne?        
if __name__ == 'main':
    dirs = []
    os.chdir(videofolder)
    videos = np.sort([fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and "temp" not in fn])

    if not len(videos):
        sys.popen('.\Extraction.py')
        videos = np.sort([fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and "temp" not in fn])

    if not len(videos):
        print('Sorry, no position/orientation files were found')
        exit()
    else:
        for vid in videos:
            os.chdir(vid)
            
            print('Calculating trajectory for', vid)
        
            poslnk = vid + '_position.csv'
            thetalnk = vid + '_bearing.csv'

            triangles = load_triangles(poslnk, thetalnk)
            display_triangles(triangles, vid)
            os.chdir('..')
    
'''
    
    
    
    
    
