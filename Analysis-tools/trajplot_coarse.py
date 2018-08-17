from posplot import *
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
        for video in videos:
            os.chdir(video)
            vid = video[:-8]
            print('Calculating trajectory for', vid)

            pos, theta = read_data(video)
            multiplot(pos, '', vid,save=False,display=True)
            
            os.chdir('..')
    
    
