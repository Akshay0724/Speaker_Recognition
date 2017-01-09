try:
  from lib.BOB import *
except:
  from lib.MFCC import *	
import lib.LPC as LPC  
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
import A
import record_input
import scipy.io.wavfile as wavfile
import thread
from tkMessageBox import *
def pred(record,Path=None):
 clf=joblib.load('clf.pkl')

 if record==0:
   thread.start_new_thread(record_input.rec,('test',False,))
   print 'Recording....'
   showinfo('Recording....', 'Press Ok To stop')
   A.a=1
 else:
   record_input.rec('test','False',Path=Path)

 while A.a == 1:
 	pass

 fs,sig=wavfile.read('test.wav')
 feature=extract(fs,signal=sig,n_ceps=20,n_filters=75)
 feature_lpc=LPC.extract(fs,sig,n_lpc=13)
 feature=np.concatenate((feature,feature_lpc),axis=1) 

 Y=clf.predict(feature)


 unique,count=np.unique(Y,return_counts=True)
 MAX_count=np.max(count)
 ind=np.where(count==MAX_count)

 predicted_person=unique[ind]

 train_score=np.fromfile('train_score.txt')


 score=train_score[predicted_person]

 score_obtained=sum(clf.predict_proba(feature)[:,predicted_person])/(len(clf.predict_proba(feature)[:,predicted_person])+0.0)

 print 'score_o = ',score_obtained,' score = ',score
 if score_obtained>=score*0.8:
 	spk=np.fromfile('Trained_Speaker.txt',dtype='|S6')
 	return 'Recognized As: ',spk[predicted_person]
 else:
    return 'Not Recognized'	

