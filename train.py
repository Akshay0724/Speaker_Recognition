
try:
  from lib.BOB import *
except:
  from lib.MFCC import *	
import os
import lib.LPC as LPC
import numpy as np
import scipy.io.wavfile as wavfile
from sklearn.svm import SVC,LinearSVC
from sklearn.externals import joblib

X=np.ones((1,33),int)
Y=np.array([[1]])
index=[]
i=0
speaker=np.fromfile('Speaker.txt',dtype='|S6')

try:
    Trained_sp=np.fromfile('Trained_Speaker.txt',dtype='|S6')
except:
    Trained_sp=np.array([],dtype='|S6')

Trained_sp=np.concatenate((Trained_sp,speaker),axis=0)

print 'To Be Trained: '
for person in Trained_sp:
    print person


speaker=np.array([],dtype='|S6')
speaker.tofile('Speaker.txt')
spr=[]
for person in Trained_sp:
    try: 
        fs,sig=wavfile.read('Silence_Removed/'+str(person)+'.wav','r')
        spr.append(person)
        feature=extract(fs,signal=sig,n_ceps=20,n_filters=75)
        feature_lpc=LPC.extract(fs,sig,n_lpc=13)
        feature=np.concatenate((feature,feature_lpc),axis=1) 
        Y=np.append(Y,[i]*feature.shape[0])
        i+=1
        X=np.concatenate((X,feature),axis=0)
    except:
    	print person+' Not in Database'
np.array(spr,dtype='S6').tofile('Trained_Speaker.txt')
Y=Y[1:]
X=X[1:,:]

clf=SVC(C=0.09,kernel='linear',probability=True)

clf.fit(X,Y)

print 'trained score:',clf.score(X,Y)





trained=[]
for i in range(i):
    print 'y = '+str(i),'sum = ',sum(Y==i),sum(clf.predict_proba(X)[:,i][Y==i])/sum(Y==i)
    trained.append(sum(clf.predict_proba(X)[:,i][Y==i])/sum(Y==i))


np.array(trained).tofile('train_score.txt')




joblib.dump(clf, 'clf.pkl') 
