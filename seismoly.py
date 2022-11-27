
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.fft import fft, fftfreq
from numpy import dot
from numpy.linalg import norm

def window(npr,dt,size,ovlp):
    '''
    Segmenta una serie temporal con un procentaje de translape
        npr  = Serie temporal 
        dt   = Delta de tiempo (1/f)
        size = Ventana de tiempo en segundos
        ovlp = Porcentaje de translape (%)
    return:
        ls   = Lista de segmentos
        ovlp = Tamaño del translape
    '''
    ls=[] #Lista con todos los segmentos
    size=size/dt #Tamaño de los segmentos
    ovlp=((100-ovlp)/100)*size #Tamaños del translape
    r0=0 #Iniciacion de los cortes
    r1=int(size)
    for _ in range(0,int(len(npr)/ovlp)): #Cortes
        ls.append(npr[r0:r1])
        r0=int(r0+ovlp)
        r1=int(r1+ovlp)
    return ls,ovlp

def descrp(arr,dt):
    '''
    Da una lista de descriptores de la señal basado en :
    Watson, L. M. (2020). Using unsupervised machine learning 
    to identify changes in eruptive behavior at Mount Etna, Italy. 
    Journal of Volcanology and Geothermal Research, 
    405, 107042.
        arr  = Ventana de tiempo
        dt   = Delta de tiempo
    return:
    Una lista de:
        arr    = Ventana de tiempo
        std    = Desviación estandar (dominio del tiempo)
        kur    = Curtosis (dominio del tiempo)
        skewt  = Asimetría (dominio del tiempo)
        skewf  = Asimetría (dominio de la frecuencia)
        pkfreq = Pico de frecuencia
        prfreq = Frecuencia en percentil 50
        *Q factor no abordado en esta aproximación hasta ahora*
    '''
    std=np.std(arr)   #Desviación estandar (dominio del tiempo)
    kur=kurtosis(arr) #Curtosis (dominio del tiempo)
    skewt=skew(arr)   #Asimetría (dominio del tiempo)
    # Dominio de la Frecuencia
    N = len(arr)      
    fft_x = fft(arr) / N  # FFT Normalizada
    freq = fftfreq(N, dt) # Recuperamos las frecuencias
    ampl=abs(fft_x[range(int(N/2))]) # Quitamos el espejo
    freq=freq[range(int(N/2))]
    skewf=skew(ampl)  #Asimetría (dominio de la frecuencia)
    try: #Pico de frecuencia
        pkfreq=freq[ampl.argmax()][0] #En dado caso que que haya 2 pico en las frecuencias, se escoje el primero
    except:
        pkfreq=freq[ampl.argmax()]
    prfreq=freq[np.where(ampl == np.percentile(ampl,50,method='closest_observation'))[0]][0] #Frecuencia en percentil 50
    return [arr,std,kur,skewt,skewf,pkfreq,prfreq]