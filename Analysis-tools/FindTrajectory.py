import os.path
import sys
import io
subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(subfolder + "/Generating_a_Training_Set")

import numpy as np

from myconfig import bodyparts, center_of_mass
from myconfig_analysis import videofolder


def leg_pair_eval(parts , i):
    '''
    Determine approximate center of mass position using the
    position of a pair of legs (either front or rear)
    Determines orientation using head, ears ,or tail
    '''

    lx = parts['ul_paw'][i][0]
    ly = parts['ul_paw'][i][1]

    rx = parts['ur_paw'][i][0]
    ry = parts['ur_paw'][i][1]
    
    # determine  position of midpoint of line connecting legs
    midx = (rx + lx) / 2
    midy = (ry + ly) / 2
    
    slope = (ry - ly) / (rx - lx)

    #pseudo code - redo
    if parts['port'][i][0]:
        reference  = 'port'
    elif parts['l_ear'][i][0]:
        reference = 'l_ear'
    elif parts['r_ear'][i][0]:
        reference = 'r_ear'
    elif parts['tail']:
        reference = 'tail'
    elif parts['nose'][i][0]:
        reference = 'nose'
    else:
        reference = 0

    if reference:
        refx = parts[reference][i][0]
        refy = parts[reference][i][1]

        sign = np.sign((rx - lx) * (refy - ly) - (ry - ly) * (refx - lx))

def compute_theta(p1x,p1y,p2x,p2y):
    '''
    Computes angle between two points using
    arctangent of slope. Onus is on user to check
    for zero-division cases
    '''
    slope = (p2y - p1y) / (p2x - p1y)
    return np.arctan(slope)
    
        

def compute_distance(p1,p2):
    '''
    For specific case where p1 and p2 are tuples of (x,y)
    '''
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1]))
        
#strongly request that center of mass

def calc_trajectory(parts):
    '''
    Calculates trajectory of head by averaging extant
    values for parts of the head 

    parts is a dictionary with part names and keys
    and (x,y) tuples as values

    Calculates heading by computing arctangents of various lines
    Current order for heading is:
        port - nose
        left ear - right ear
        port - left ear 
        port - right ear 
        linear interpolation
        

    
    '''
    frames = len(parts[bodyparts[0]]) # len shouldn't differ
    position = [(0,0)] * frames
    bearing = [0] * frames
    head_parts = ['port', 'nose','l_ear','r_ear']

    unsolved_pos = []
    unsolved_bearing = []
    
    for i in range(frames):
        posx = []
        posy = []
        for part in head_parts:
            if parts[part][i][0]: # check if a value was computed
                posx.append(parts[part][i][0])
                posy.append(parts[part][i][1])
        if len(posx):
            position[i] = (np.mean(posx), np.mean(posy))
        else:
            unsolved_pos.append(i)

 
                
        portx = parts['port'][i][0] * (parts['port'][i][0] > 0)
        porty = parts['port'][i][1] 

        nosex = parts['nose'][i][0]  * (parts['nose'][i][0] > 0)
        nosey = parts['nose'][i][1]

        learx = parts['l_ear'][i][0] * (parts['l_ear'][i][0] > 0)
        leary = parts['r_ear'][i][1]

        rearx = parts['r_ear'][i][0] * (parts['r_ear'][i][0] > 0)
        reary = parts['r_ear'][i][1]

        slopes = []

        left_ear_theta = -0.4424
        '''
        more precision: -0.44241058216837614
        Computed by averaging location of left ear
        at indices 30 and 35 in /183856c/l_ear.csv.
        The mouse in these frames was facing near-vertical
        '''
        right_ear_theta = 0.6872
        '''
        more precision: 0.6871831742241893
        Computed by averaging location of right ear
        at indices 30 and 35 in /183856c/r_ear.csv.
        The mouse in these frames was facing near-vertical
        '''
        #handle slope 
        if portx and nosex:
            try:
                slope = (porty - nosey) / (portx - nosex)
            except ZeroDivisionError:
                slope = 1e5
            bearing[i] = np.arctan(slope)
        elif learx and rearx:
            try:
                slope = (reary - leary) / (rearx - learx)
            except ZeroDivisionError:
                slope = 1e5
            bearing[i] = np.arctan(-1 / slope)
        elif learx and portx:
            raw_theta = compute_theta(learx,leary,portx,porty)
            bearing[i] = left_ear_theta - raw_theta - np.pi / 2 
        elif rearx and portx:
            raw_theta = compute_theta(rearx,reary,portx,porty)
            bearing[i] = right_ear_theta - raw_theta - np.pi / 2 
        else:
            unsolved_bearing.append(i)
            continue
    interpolate_missing(position, unsolved_pos)
    interpolate_missing(bearing, unsolved_bearing)
    return position, bearing
           
            
def get_not_missing(start, missing_lst, sign, lstlen):
    '''
    Finds some index for which a value exists 
    missing_lst is a list of indices for which values are
    missing. Start is the starting index for the search (most
    useful if start is in missing_lst), sign is 1 or -1, and
    determines the direction the search progresses in.
    lstlen is the length of the original list
    '''
    value = start
    while value in missing_lst:
        if value != (sign > 0) * lstlen - 1:
            value += sign
        else:
            break
    return not_missing
    
def interpolate_missing(lst, indices):
    '''
        Performs linear interpolation of missing data from
        a lst, and and a list of indices for which interpolation
        is needed.
    '''
    if len(indices) == len(lst):
        print('There are no solved points')
        return

    # best not to iterate over a dynamically changing list 
    static_missing_lst = indices.copy()

    for index in static_missing_lst:
        '''
        Iterate in the given direction until some solved index is
        reached.
        '''
        lthreshold = get_not_missing(index, indices, -1, len(lst))
        rthreshold = get_not_missing(index, indices, 1, len(lst))
                
        l_threshold_val = lst[lthreshold]
        r_threshold_val = lst[rthreshold]
        
        #assumes that only one doesn't exist
        #handle boundary conditions
        near_threshold = lthreshold
        
        if not l_threshold_val:
            lthreshold = rthreshold
            rthreshold = get_not_missing(rthreshold + 1, indices, 1, len(lst))
        elif not r_threshold_val:
            rthreshold_val = lthreshold
            lthreshold_val = get_not_missing(lthreshold - 1, indices, -1, len(lst))
            near_threshold = rthreshold
            
        l_threshold_val = lst[lthreshold]
        r_threshold_val = lst[rthreshold]
        near_threshold_val = lst[near_threshold]
        
        # this is assuming no boundary condition
        delta = r_threshold_val - l_threshold_val
        
        fraction = (index - near_threshold) / (rthreshold - lthreshold)
        itp_value = near_threshold_val + fraction * delta
        lst[index] = itp_value
        indices.pop(index)


def load_parts():
    '''
    Loads body part location data from csvs 
    '''
    bparts = {}

    for part in bodyparts:
        try:
            with open(part + '.csv') as f:
                lines = f.readlines()
                locations = []

                for line in lines:
                    x = float(format(float(line.split(',')[0]),'.3f'))
                    y = float(format(float(line.split(',')[1].split('\n')[0]),'.3f'))
                    locations.append((x,y))
                bparts[part] = locations
                
        except FileNotFoundError:
            print(part + ' location file not found.')
    return bparts
            
    
dirs = []
os.chdir(videofolder)
videos = np.sort([fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and "temp" not in fn])

if not len(videos):
    sys.popen('.\Extraction.py')
    videos = np.sort([fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and "temp" not in fn])

if not len(videos):
    print('Sorry, no subdirectories were found. Please recheck Extraction.py')
    exit()
else:
    for vid in videos:
        os.chdir(vid)
        parts = load_parts()
        position, bearing = calc_trajectory(parts)

        # Output to file 
        with open(vid + '_position.csv') as f:
            for pos in position:
                f.write(pos + '\n')

        with open(vid + '_bearing.csv') as f:
            for bear in bearing:
                f.write(bearing + '\n')
    
