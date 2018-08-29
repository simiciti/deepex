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
            output = tc_segmented(pos, time_circle, '', segments[vid]['segments'], vid,
                      segments[vid]['labels'], aggregate=True, verbose=False)

            for otpt in output:
                lbl = otpt[0]
                times = otpt[1]
                circles = otpt[2]
                
                with open(vid + '_{0}_circle_times.txt'.format(lbl), 'w') as f:
                    f.write('ACTIVE: ' + str(ACTIVE) + '\n')
                    for i in range(len(circles[0])):
                        f.write('{0} px '.format(circles[0][i]) + ' ' +
                                '{:.3f} seconds\n'.format(times[0][i]))
                    f.write('MOVED: ' + str(MOVED) + '\n')
                    for i in range(len(circles[1])):
                        f.write('{0} px '.format(circles[1][i]) + ' ' +
                            '{:.3f} seconds\n'.format(times[1][i]))
                    
                    
                
            '''
            segmented(pos, multiplot, '', segments[vid]['segments'], vid,
                      segments[vid]['labels'], aggregate=True, verbose=True,
                      save=True, display=False, stim=True, tag='s')
            '''
            #multiplot(pos, '', vid,save=True,display=False)
            
            os.chdir('..')
    
    
