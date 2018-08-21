from posplot import *
from optogenetic_helper import *
if __name__ == '__main__':
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
        segments = perform_correspondence('../p2p/camera correspondence.txt')
        for video in videos:
            os.chdir(video)
            vid = video[:-8]
            print('Plotting fine trajectory for', vid)

            pos, theta = read_data(video)
            segmented(pos, multiplot, '', segments[vid]['segments'], vid,
                      segments[vid]['labels'], aggregate=True, verbose=True,
                      save=True, display=False, stim=True, tag='s')
            
            #multiplot(pos, '', vid,save=True,display=False)
            
            os.chdir('..')
    
    
