import record_input
import thread,time
import A
import numpy as np

print '1. Enrollment\n2. Training\n3. Recognition\n'

i=int(input())

if i==1:
	print 'Enter Name: '
	name=raw_input()
	thread.start_new_thread(record_input.rec,(name,True,))
	a=raw_input()
	A.a=1
	speaker=np.fromfile('Speaker.txt',dtype='|S6')
	speaker=np.concatenate((speaker,[name]),axis=0)
	speaker=np.array(speaker,dtype='|S6')
	speaker.tofile('Speaker.txt')
	while A.a == 1:
		pass
if i==2:
   import train		