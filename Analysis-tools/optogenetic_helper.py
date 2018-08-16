import os.path
import sys
import io

import numpy as np

subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)

from myconfig import bodyparts, center_of_mass
from myconfig_analysis import videofolder


def lbl_decode(lbl):
    '''
    Decodes segmentation label
    '''
    if lbl == 's':
        return 'start'
    elif lbl == 'a':
        return 'active'
    elif lbl == 'm':
        return 'moved'
    elif lbl == 'd':
        return 'inactive'
    else:
        raise ValueError('Supplied label has an invalid value: {0}. Should be \
s, a, m or d'.format(lbl))
        
        
def load_segmentation():
    '''
    Loads segmentation data and labels for
    experiments from text files in experiment subdirectories in videofolder
    and returns a dictionary arranged as
    segmentations[experiment]
        [segments]
        [labels].
    '''
    cwd = os.getcwd()
    os.chdir(videofolder)

    #find experiment folders 
    folders = [fn for fn in os.listdir(os.curdir) if os.path.isdir(fn) and "temp" not in fn]

    segmentations = {}

    # iterate through folders
    for folder in folders:
        try:
            segments = []
            labels = []

            #get the experiment name 
            fol = folder.split('_extract')[0]
            
            #parse the segmentation file
            with open(folder + '/' + fol + '_segments.txt') as f:
                for line in f.readlines():
                    if len(line) != 1: #handles case of empty lines
                        labels.append(lbl_decode(line.split(' ')[0]))
                        segments.append(int(line.split(' ')[1]))

                #remove the stimulus appearance frame
                labels.pop(0)
                segments.pop(0)
            
            #add to dictionary
            segmentations[fol] = {}
            segmentations[fol]['segments'] = segments
            segmentations[fol]['labels'] = labels
    
        except IOError:
            #handles the file not found case
            print('File not found. Unfortunately, the epoch file can\'t be created \
                programmatically. It must be manually created.')
            
    if not len(segmentations):
        # It was unable to find segmentation data for all files 
        print('Error. Segmentation dictionary is empty. Aborting')
        os.chdir(curdir)
        exit()
    else:
        os.chdir(cwd)
        return segmentations

def perform_correspondence(corrfile):
    '''
    Calculates correspondence figure between different cameras using
    properly-formatted file. 
    '''
    corrs = []
    try:
        with open(corrfile) as f:
            for line in f.readlines:
                if len(line) == 1:
                    continue
                if line[0] == '#': #commented out lines
                    continue
                parsed = line.split(' ')
                #append [experiment, webcam frame, IR camera frame]
                corrs.append((parsed[0],int(parsed[2]) - int(parsed[3])))
    except FileNotFoundError:
        print('correspondence file invalid. Exiting')
        exit()

    segs = load_segmentation()

    if not len(corrs): #the correlations list is empty
        raise Exception('No correspondences were found. Exiting.')
        exit()

    #convert the webcam frame to the IR camera frame using the correspondence
    for corr in corrs:
        for i in range(len(segs[corr[0]]['segments'])):
            segs[corr[0]]['segments'][i] -= corr[1]

    return segs 
        
        
            
                
        
