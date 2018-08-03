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

from tqdm import tqdm
'''
def leg_pair_eval(parts , i):
    
    Determine approximate center of mass position using the
    position of a pair of legs (either front or rear)
    Determines orientation using head, ears ,or tail


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
'''
def angle_clamp(theta):
    '''
    Ensures theta is between 0 and 2π in value.
    '''
    if theta < 0:
        return theta + 2 * np.pi
    elif theta > 2 * np.pi:
        return theta - 2 * np.pi
    else:
        return theta

def compute_theta(p1x,p1y,p2x,p2y, cw=False, theta_ref=(np.pi/2)):
    '''
    Computes angle between two points p1 and p2, which are tuples of
    form (x,y).
    p1 should be the point furthest along the direction of the line segment.
    For example, the line connecting the center of the head to the nose
    is parallel to the head orientation, but the nose is further in
    that direction, so it would be p1 and the center of the head would
    be p2.

    cw (short for clockwise) is a boolean describing whether to proceed
    in a clockwise(p1 furthest in given direction) or counterclockwise
    (p2 furthest in given direction) manner. The center of head- left ear
    line would be cw=False, while the center of head-right ear line would
    be cw=True.

    theta_ref is a reference angle. It is used when the points provided
    define a line oriented at some angle theta_ref (clockwise
    if y axis points upwards, counterclockwise if y axis points downward)
    to the line whose angle theta is being calculated.
    
    For direct slope (or equivalent when slope is undefined),
    use the default values for cw and theta_ref.
    '''
    try:
        slope = (p2y - p1y) / (p2x - p1x)
        sign = 1 - 2 * (cw == (p2x > p1x))
        return angle_clamp(sign * (np.pi / 2) + np.arctan(slope) + theta_ref)
    # slope points vertical
    except ZeroDivisionError:
        return angle_clamp((cw == (p2y < p1y)) * np.pi + theta_ref)
    
        

def compute_distance(p1,p2):
    '''
    For specific case where p1 and p2 are tuples of (x,y)
    '''
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calc_trajectory(parts,do_ear_port = False):
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

    Orientation is in radians

    do_ear_port allows attempts to determine orientation using
    the port and one ear. As this distance is not necessarily constant
    in the photograph (and the reference angle varies), it really
    shouldn't be used.
    '''
    
    frames = len(parts[bodyparts[0]]) # len shouldn't differ
    position = [(0,0)] * frames
    bearing = [-100] * frames
    head_parts = ['port', 'nose','l_ear','r_ear']

    unsolved_pos = []
    unsolved_bearing = []
    for i in tqdm(range(frames)):
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
        leary = parts['l_ear'][i][1]

        rearx = parts['r_ear'][i][0] * (parts['r_ear'][i][0] > 0)
        reary = parts['r_ear'][i][1]


        l_ref = 3.3657
        '''
        Left reference angle 
        more precision: 3.365715279788638 #just for reference, don't use
        Computed by averaging location of left ear
        at indices 30 and 35 in /183856c/l_ear.csv.
        The mouse in these frames was facing near-vertical
        '''
        r_ref = 2.6239
        '''
        Right reference angle
        more precision: 2.6238766262729953 #just for reference, don't use
        Computed by averaging location of right ear
        at indices 30 and 35 in /183856c/r_ear.csv.
        The mouse in these frames was facing near-vertical
        '''

        #handle bearing
        if portx and nosex:
            # get direct angle
            bearing[i] = compute_theta(nosex, nosey, portx, porty)
        elif learx and rearx:
            # get angle of orthogonal line 
            ortho = compute_theta(learx, leary, rearx, reary)
            # convert to actual orientation 
            bearing[i] = angle_clamp(ortho + np.pi / 2)
        elif learx and portx and do_ear_port:
            # cw = False as left ear is counterclockwise from orientation
            bearing[i] = compute_theta(learx,leary,portx,porty, False,l_ref)
        elif rearx and portx and do_ear_port:
                        
            bearing[i] = compute_theta(rearx,reary,portx,porty, True, r_ref)
        else:
            unsolved_bearing.append(i)
            
    print('Unsolved position fraction:', len(unsolved_pos) / len(position))
    print('Unsolved bearing fraction:', len(unsolved_bearing) / len(bearing))
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
    return value
    
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
        
        # tuples are handled separately
        fraction = (index - near_threshold) / (rthreshold - lthreshold)
        if isinstance(l_threshold_val, tuple):
            delta = tuple(np.subtract(r_threshold_val, l_threshold_val))
        
            
            adj_delta = tuple(np.multiply(fraction, delta))
            itp_value = tuple(np.add(near_threshold_val, adj_delta))
        else:
            #it is a float or int
            delta = r_threshold_val - l_threshold_val
            itp_value = near_threshold_val + fraction * delta
            
        lst[index] = itp_value
        indices.remove(index) # wait, pop value rather than index


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
videos = np.sort([fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and "_extract" in fn])

if not len(videos):
    sys.popen('.\Extraction.py')
    videos = np.sort([fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and "_extract" in fn])

if not len(videos):
    print('Sorry, no subdirectories were found. Please recheck Extraction.py')
    exit()
else:
    for vid in videos:
        os.chdir(vid)
        parts = load_parts()
        
        print('Calculating trajectory for', vid)
    
        poslnk = vid[:-8] + '_position.csv'
        bearlnk = vid[:-8] + '_bearing.csv'

        if (os.path.isfile(poslnk)):
            print('Position file already exists at ' + poslnk + '. Skipping')
            os.chdir('..')
            continue
        if (os.path.isfile(bearlnk)):
            print('Bearings file already exists at ' + bearlnk + '. Skipping')
            os.chdir('..')
            continue
        

        position, bearing = calc_trajectory(parts)

        print('Validating positions...')
        # Check if unsolved indices exist
        if len([v for v in position if not v[0]]):
            checkfail = input('Position Validation failed. Exit? Y/n')
            if checkfail.lower() == 'y':
                exit()

        print('Validating bearings...')
        # Check if unsolved indices exist
        if len([v for v in bearing if v < -20]):
            checkfail = input('Bearing Validation failed. Exit? Y/n')
            if checkfail.lower() == 'y':
                exit()
                
        # Output to file
        with open(poslnk, 'w') as f:
            for pos in position:
                f.write(str(pos[0])+ ',' + str(pos[1]) + '\n')
            print('Position data serialized to ' + poslnk)

        
        with open(bearlnk, 'w') as f:
            for bear in bearing:
                f.write(str(bear) + '\n')
            print('Bearing data serialized to ' + bearlnk)
        os.chdir('..')
    
