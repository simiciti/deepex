import glob
import pandas as pd
import numpy as np

import os.path
import sys
subfolder = os.getcwd().split('Analysis-tools')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "/pose-tensorflow/")
sys.path.append(subfolder + "/Generating_a_Training_Set")

import auxiliaryfunctions
import io

#from moviepy.editor import VideoFileClip
from myconfig_analysis import videofolder, pcutoff, videotype, resnet, \
     Task, date, shuffle, trainingsiterations
from tqdm import tqdm


# Name for scorer based on passed on parameters from myconfig_analysis. Make sure they refer to the network of interest.
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)



def ExtractPositions(dataframe, folder=True):
    '''
    Extracts body part positions to labeled files
    if folder is true, a new folder named for the file is created
    else extracted body part position file names include original video file
    defaults to 0,0 if the likelihood was below the cutoff
    '''
    nframes = len(dataframe.index)
    scorer=np.unique(dataframe.columns.get_level_values(0))[0]
    bodyparts = list(np.unique(dataframe.columns.get_level_values(1)))

    print(pcutoff)
    for bodypart in bodyparts:
        print('extracting ' + bodypart)
        if folder:
            auxiliaryfunctions.attempttomakefolder(vname + '_extract')
            file = open(vname + '_extract/' + bodypart + '.csv','w')
        else:
            file = open(vname +  '_' + bodypart + '.csv', 'w')
        for index in tqdm(range(nframes)):
            if dataframe[scorer][bodypart]['likelihood'].values[index] > pcutoff:
                line = str(dataframe[scorer][bodypart]['x'].values[index])
                line += ','
                line += str(dataframe[scorer][bodypart]['y'].values[index]) + '\n'
                
            else:
                line = '0,0\n'
            file.write(line)
        file.close()
    print('extraction completed')
        
        
        
    
    
##### Included from DeepLabCut/Analysis/MakingLabeledVideo.py

os.chdir(videofolder)
videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn) and ("labeled" not in fn)])

print("Starting ", videofolder, videos)
for video in videos:
    vname = video.split('.')[0]
    if os.path.isdir(vname + '_extract'):
        if len(glob.glob(vname + '_extract/*.csv')):
            print(vname + ' data already extracted. Skipping.')
            continue
    print("Loading ", video, "and data.")
    dataname = video.split('.')[0] + scorer + '.h5'
    try: # to load data for this video + scorer
        Dataframe = pd.read_hdf(dataname)
        clip = VideoFileClip(video)
        ExtractPositions(clip,Dataframe)
    except FileNotFoundError:
        datanames=[fn for fn in os.listdir(os.curdir) if (vname in fn) and (".h5" in fn) and "resnet" in fn]
        if len(datanames)==0:
            print("The video was not analyzed with this scorer:", scorer)
            print("No other scorers were found, please run AnalysisVideos.py first.")
        elif len(datanames)>0:
            print("The video was not analyzed with this scorer:", scorer)
            print("Other scorers were found, however:", datanames)
            print("Extracting from:", datanames[0]," instead.")

            Dataframe = pd.read_hdf(datanames[0])
            #clip = VideoFileClip(video)
            ExtractPositions(Dataframe)
            

                        
