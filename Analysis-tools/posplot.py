import os.path
import sys
import io

import matplotlib.pyplot as plt
import matplotlib.path as mpath

from moviepy.editor import VideoFileClip
subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(subfolder + "/Generating_a_Training_Set")

import numpy as np

from tqdm import tqdm

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
    


    
def heatmap(pos, folder, imname='', export=True, save=True):
    '''
    Converts trajectory data to pixel-based heatmap
    agnostic of whether stimulus was present or mouse was activated.
    '''

    # Find the limits in the x and y directions; These will be the boundaries
    # of the frame 
    [xlim, ylim] = np.nanmax(pos, axis=0)
    xlim, ylim = int(xlim) + 20, int(ylim) + 20

    # Initialize cells
    cells = np.zeros((xlim,ylim))


    #placeholder value - set cell to 1 if the mouse was at it
    increment = 1
    for i in tqdm(range(len(pos))):
        cells[int(pos[i][0])][int(pos[i][1])] += increment
    
    fig, ax = plt.subplots()

    #transpose because it inverts the display (but not the actual graph)
    im = ax.imshow(cells.T, cmap='Y1Gn')
    cbar = ax.figure.colorbar(im,ax=ax,cmap='YlGn')

    if not export:
        plt.title(imname)

    if save:
        plt.savefig(folder + '/' + imname + '_heatmap.png')
    if export:
        return im #I think that's correct


def segmented(pos, function, folder, seg, imname='', labels=None):
    '''
    Provides an easy interface for segmenting the position data and
    applying any properly parameterized function to the data.
    '''

    for i in range(1, len(seg)):
        cur = seg[i]
        lo = seg[(i - 1) * (( i - 1) > 0)] * ((i - 1) > 0)

        epoch = pos[lo: cur]
        lbl = labels[i - 1] if labels else ''
        
        function(epoch, folder, lo + '-' + cur + ' ' + lbl)
    

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

    
def plot_trajectory(pos, folder, name='', adj_name='',export=True,save=True):
    '''
    Be a right-sider. Connect the cases.
    opacity: 0.2
    
    '''
    [xlim, ylim] = np.nanmax(pos, axis=0)
    xlim, ylim = int(xlim) + 20, int(ylim) + 20

    opacity = 0.2
    
    plotted = plt.plot(pos[:,0],pos[:,1], alpha=opacity)

    if not export:
        plt.title(name)
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        plt.xlabel('pixels')
        plt.ylabel('pixels')

    file = adj_name if adj_name else name
    if save:
        plt.savefig(folder + '/' + file + '_traj.png')
    if export:
        return plotted

#def plot_trajectory_seg():
#    return None 
    
def kde(pos, folder, name='', export=True, save=True):
    '''
    Plots a kernel density estimate for the position data
    '''

    [xlim, ylim] = np.nanmax(pos, axis=0)
    xlim, ylim = int(xlim) + 20, int(ylim) + 20
    
    #create cubehelix colormap
    cmap = sns.cubehelix_palette(start=3, light=1, as_cmap=True)

    #actual plot
    s = sns.kdeplot(pos[:,0], pos[:,1], cmap=cmap, shade=True)


    if not export:
        #data 
        plt.title(name)
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        plt.xlabel('pixels')
        plt.ylabel('pixels')

    if save:
        plt.savefig(folder + '/' + name + '_kde.png')
    if export:
        return s
    
def multiplot(pos, name):
    '''
    Plots multiple representations of position data for a single experiment
    '''

    [xlim, ylim] = np.nanmax(pos, axis=0)
    xlim, ylim = int(xlim) + 20, int(ylim) + 20
    
    f, axes = plt.subplots(3,1, figsize=(9,9), sharex=True, sharey=True)

    functions = [heatmap, kde, plot_trajectory]

    #create grid of subplots 
    for ax, s in zip(axes.flat, np.linspace(0,3,3)):
        functions[s](pos, name)
        ax.set(xlim=(0,xlim), ylim=(0,ylim))

    plt.title(name)
    f.tight_layout

        



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
    
    
    
    
    
