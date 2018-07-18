import glob, io, numpy as nu


with open('port.csv') as f:
    port = f.readlines()

with open('l_ear.csv') as f:
    lear = f.readlines()

with open('r_ear.csv') as f:
    rear = f.readlines()

with open('nose.csv') as f:
    nose = f.readlines()
csv = glob.glob('*.csv')
for c in csv:
    with open(c, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            try:
                if int(line[0]):
                    count += 1
            except ValueError:
                print(line)
    print(c, count / len(lines))


troika = 0
np = 0
pl = 0
pr = 0
ntroi = 0
nl = 0
nr = 0
lr = 0
for i in range(len(nose)):
     if int(port[i][0]) and int(lear[i][0]) and int(rear[i][0]):
             troika += 1
     if int(port[i][0]) and int(nose[i][0]):
             np += 1
     if int(port[i][0]) and int(lear[i][0]):
             pl += 1
     if int(port[i][0]) and int(rear[i][0]):
             pr += 1
     if int(nose[i][0]) and int(lear[i][0]) and int(rear[i][0]):
             ntroi += 1
     if int(nose[i][0]) and int(lear[i][0]):
             nl += 1
     if int(nose[i][0]) and int(rear[i][0]):
             nr += 1
     if int(lear[i][0]) and int(rear[i][0]):
             lr += 1
print('\n\n')
print('troika', troika / len(nose))
print('np', np / len(nose))
print('pl', pl /len(nose))
print('pr', pr / len(nose))
print('ntroi',ntroi/len(nose))
print('nl',nl/len(nose))
print('nr',nr/len(nose))
print('lr', lr/ len(nose))

solved = [0] * len(nose)

for i in range(len(nose)):
     #if int(port[i][0]) and int(lear[i][0]) and int(rear[i][0]):
     #        solved[i] += 1
     if int(port[i][0]) and int(nose[i][0]):
             solved[i] += 1
     if int(port[i][0]) and int(lear[i][0]):
            solved[i] += 1
     if int(port[i][0]) and int(rear[i][0]):
             solved[i] += 1
     #if int(nose[i][0]) and int(lear[i][0]) and int(rear[i][0]):
     #        solved[i] += 1
     #if int(nose[i][0]) and int(lear[i][0]):
     #        solved[i] += 1
     #if int(nose[i][0]) and int(rear[i][0]):
     #        solved[i] += 1
     if int(lear[i][0]) and int(rear[i][0]):
             solved[i] += 1

print('mean',nu.mean(solved))
print('std', nu.std(solved))

mu = [fr for fr in solved if not fr]
print('fraction unsolved' , len(mu) / len(solved))



