from lib.Remove_white import *
import scipy.io.wavfile as wavfile
import alsaaudio, wave, numpy,A


def rec(name,Is_train,Path=None):
  if Path is None:
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
    inp.setchannels(1)
    inp.setrate(44100)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(1024)

    w = wave.open('test.wav', 'w')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(44100)

    while A.a == 0:
      l, data = inp.read()
      a = numpy.fromstring(data, dtype='int16')
      #print numpy.abs(a).mean()
      w.writeframes(data)
    fs,signal=wavfile.read('test.wav')
  else: 
  	fs,signal=wavfile.read(Path)
  signal=remove_silence(fs,signal)#,perc=0.25)#,frame_duration = 0.01,frame_shift = 0.005)

  if Is_train == True:
     wavfile.write('Silence_Removed/'+name+'.wav',fs,signal)
  else:
     wavfile.write('test.wav',fs,signal)	 
  A.a=0 

