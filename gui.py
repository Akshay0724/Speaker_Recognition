import record_input
import thread,time
import A
import numpy as np
import Tkinter as tk
from tkMessageBox import *
from tkFileDialog   import askopenfilename      
import predict
root=tk.Tk()
fr=tk.Frame(root)
text=tk.Entry(fr)


def callback():
  print text.get()
  if askyesno('Record Mode', 'Yes to Record else No to Import Wavfile'):
    name=text.get()
    thread.start_new_thread(record_input.rec,(name,True,))
    showinfo('Recording....', 'Press Ok To stop')
    A.a=1
    speaker=np.fromfile('Speaker.txt',dtype='|S6')
    speaker=np.concatenate((speaker,[name]),axis=0)
    speaker=np.array(speaker,dtype='|S6')
    speaker.tofile('Speaker.txt')
    while A.a == 1:
	  pass
    
  else:
    name=text.get()
    Path=askopenfilename() 
    record_input.rec(name,True,Path=Path)
    speaker=np.fromfile('Speaker.txt',dtype='|S6')
    speaker=np.concatenate((speaker,[name]),axis=0)
    speaker=np.array(speaker,dtype='|S6')
    speaker.tofile('Speaker.txt')
def Train():
	import train
	showinfo('','Model Updated:')

def Predict():
    record=1
    if askyesno('Record Mode', 'Yes to Record else No to Import Wavfile'):
       message=predict.pred(0)
    else:
       Path=askopenfilename()
       message=predict.pred(1,Path=Path)
    showinfo('Result',message)      
       
def main():
  
  button = tk.Button(fr, text='Enroll', width=25, command=callback)
  button.pack(padx=5,pady=5,side='left')
  
  text.config(font=('times', 18, 'italic'))
  text.pack(side='right', expand='yes', fill='x')
  fr.pack()
  #text.tag_configure('big', font=('Arial', 12, 'bold', 'italic'))
  #text.tag_bind('follow', '<1>', lambda e, t=text: t.insert(END, "Not now, maybe later!"))
  #text.insert(END,'','big')
  #text.insert(END,'\nWilliam Shakespeare\n', 'big')
  
  button1 = tk.Button(root, text='Train', width=25, command=Train)
  button1.pack(padx=5,pady=5)
  
  button1 = tk.Button(root, text='Predict', width=25, command=Predict)
  button1.pack(padx=5,pady=5)


  root.mainloop()
  

  





if __name__=='__main__':
   main()   
