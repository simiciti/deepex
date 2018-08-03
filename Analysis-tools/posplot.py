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
    
    
def data_to_heatmap(pos, folder):
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
        cells[int(pos[i][0])][int(pos[i][1])] = increment
    
    fig, ax = plt.subplots()

    #transpose because it inverts the display (but not the actual graph)
    im = ax.imshow(cells.T, cmap='Y1Gn')
    cbar = ax.figure.colorbar(im,ax=ax,cmap='YlGn')
    
    plt.savefig(folder + '_heatmap.png')

def distance_to_origin(pos, folder, fps=30):
    '''
    Graphs distance to origin as a function of time elapsed
    '''

    # Get the distance from the origin at each frame
    distances = distance(0, 0, pos[:,0], pos[:,1])
    
    frames = np.arange(len(pos))

    fig, ax = plt.subplots()

    im = ax.imshow([distances,frames])
    plt.savefig(folder + '_origin.png')

    

    


def bearing_sunspot(bearing, title):
    '''
    convert to int
    count frequency of each orientation
    create 1deg arc at base radius
    create 1deg arc scaled radius
    add labels at k* n.pi / 4
    
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
    
    
    
    
    
