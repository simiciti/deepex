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
            print('Plotting coarse circle times for', vid)

            pos, theta = read_data(video)
            #multiplot(pos, '', vid,save=True,display=False, stim=True, tag='s')
            ACTIVE = (381, 501)
            MOVED = (858, 501)
            times = []
            circles = []
            for spot in (ACTIVE, MOVED):
                time, radii = time_circle(pos, spot)
                times.append(time)
                circles.append(radii)
            
            with open(vid + '_circle_times.txt', 'w') as f:
                f.write('ACTIVE: ' + str(ACTIVE) + '\n')
                for i in range(len(circles[0])):
                    f.write('{0} px '.format(circles[0][i]) + ' ' + '{:.3f} seconds\n'.format(times[0][i]))
                f.write('MOVED: ' + str(MOVED) + '\n')
                for i in range(len(circles[1])):
                    f.write('{0} px '.format(circles[1][i]) + ' ' + '{:.3f} seconds\n'.format(times[1][i]))
            
            os.chdir('..')
    
    
