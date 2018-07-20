import os.path
import sys
import io
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(subfolder + "/Generating_a_Training_Set")

import numpy as np

from myconfig import bodyparts, center_of_mass
from myconfig_analysis import videofolder



def gen_triangle(x, y, bearing,dist=77):
    '''
    Generates an equilateral triangle centered around the given (x, y) point,
    oriented at the given bearing (in radians)
    '''
    front = np.asarray(x, y - 77)
    left = np.asarray(x - 77 * np.cos(np.pi / 3), y + 77 * np.sin(np.pi / 3))
    right = np.asarray(x + 77 * np.cos(np.pi / 3), y + 77 * np.sin(np.pi / 3))

    points = np.asarray([front, left, right])
    #inverting owing to inverted image y-axis, maybe not necessary
    transform = np.asarray([[np.cos(bearing), -np.sin(bearing)],
                            [np.sin(bearing), np.cos(bearing)]])
    return transform * points

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
    

    
    
    
    
    
