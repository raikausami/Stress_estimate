import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import csv

class MotherWavelet(object):

    def __init__(self, wavelet):
        self.wavelet = wavelet

    def __call__(self, wid, a):
        x = np.arange(wid) - int(wid / 2)
        return self.wavelet(x/a)

def gabor(t,a=8): #論文参照 aは周波数の逆数？
        k=np.exp(-1/2*(a**2))
        c=(1+np.exp(-(a**2))-2*np.exp(-3/4*(a**2)))**(-1/2)
        real=c*((np.pi)**(-1/4))*(np.exp((-1/2)*(t**2)))*(np.cos(a*t)-k)
        comp=c*((np.pi)**(-1/4))*(np.exp((-1/2)*(t**2)))*(np.sin(a*t)-k)
        return real
    #return real,comp

def cwt(x, mw, A): #Aは長いほどいいのかも
    y = []
    lf=[]
    hf=[]
    for a in A:
        wave = mw(min(10 * a, len(x)), a)
        s = np.convolve(x, wave, mode='same')
        y.append(s / np.abs(a)**0.5)
    return np.array(y)


csv_file = open("test2_410-510.csv","r",encoding="ms932",errors="",newline="")
f = csv.reader(csv_file,delimiter = ",",doublequote = True, lineterminator = "\r\n",quotechar='"',skipinitialspace=True)

ecg = []
fecg = []
header = next(f)
maximal_y = []

for row in f:
    ecg=row[2].split(":")
    fecg.append(float(ecg[0]))
    fecg.append(float(ecg[1]))
    fecg.append(float(ecg[2]))
    fecg.append(float(ecg[3]))
    fecg.append(float(ecg[4]))
    fecg.append(float(ecg[5]))
    fecg.append(float(ecg[6]))
    fecg.append(float(ecg[7]))


x = np.linspace(0,len(fecg)*0.005,len(fecg))

i=0

plt.plot(x,fecg)

plt.pause(3)

lf = []
hf= []
stress = []
widths = np.arange(4, 500)
cwtmatr = cwt(fecg, MotherWavelet(gabor), widths)
for t in range(0,len(cwtmatr[0])):
    lf.append(sum(cwtmatr[:12,[t]]))
    hf.append(sum(cwtmatr[13:,[t]]))
print(lf)
print(hf)
#for j in range(0,len(lf)-1):
for j in range(0,len(lf)):
    print(lf[j]/hf[j])
    stress.append(lf[j]/hf[j])
plt.imshow(cwtmatr, extent=[0, x[len(x)-1], -500, 500], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

plt.show()

plt.plot(x,stress)
plt.show()
