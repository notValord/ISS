# scipy - potreba normalizace
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as ss

INPUT = "../audio/xmolna08.wav"
RAM_SIZE = 1024
PRE_SIZE = 512

f1 = 850
f2 = 1700
f3 = 2550
f4 = 3400


def plot(fs, data):
    t = np.arange(data.size) / fs
    plt.figure(figsize=(6, 3))
    plt.plot(t, data)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukový signál')

    plt.tight_layout()


def plot_dft(fs, data):
    data = data[0:(data.size//2)]
    fs = fs/2
    t = np.arange(0, fs, fs/data.size)
    plt.figure(figsize=(6, 3))
    plt.plot(t, data)
    plt.gca().set_xlabel('f[Hz]')
    plt.gca().set_title('DFT')

    plt.tight_layout()


def take_input():
    fs, data = wavfile.read(INPUT)
    data = data / 2 ** 15 # normalizacia
    print("Vzorkovacia frekvencia: ", end=" ")
    print(fs)
    print("Pocet vzorkov: ", end=" ")
    print(data.size)
    print("Dlzka nahravky: ", end=" ")
    print(data.size/fs)
    print(data.min(), data.max())
    plot(fs, data)
    return fs, data


def predrobenie(data, fs):
    average = np.average(data)
    #print(average)
    data = data - average
    data = data/(abs(data).max())
    ramce = list()
    for i in range((data.size-RAM_SIZE)//PRE_SIZE):
        ramce.append(data[i*PRE_SIZE:i*PRE_SIZE+RAM_SIZE])
    plot(fs, ramce[30])
    return ramce


def make_base():
    bases = list()
    for k in range(RAM_SIZE):
        if k == 0:  # specialny pripad, ktory sposobuje chybu pri np.arange, prve pole su nuly
            tmp = np.zeros(RAM_SIZE)
        else:
            max = 2*math.pi*k  # maximum, ktore nenastane (n/N s vykratili)
            step = (2*math.pi*k)/RAM_SIZE  # krok medzi jednotlivymi hodnotami
            tmp = np.arange(0, max, step)  # vytvorenie bazy pre dane k
        tmp = tmp * (-1j)   # pridanie komplexnej jednotky
        bases.append(np.exp(tmp))   # vytvorenie komplexnej exponencialy
    return bases


def dft(data, fs):
    bases = make_base()
    out = np.array([])
    for k in range(RAM_SIZE):
        out = np.append(out, np.matmul(data, bases[k]))
    plot_dft(fs, np.abs(out))
    print(np.allclose(out, np.fft.fft(data)))


def spektogram(data, fs):
    f, t, sgr = ss.spectrogram(data, fs, nperseg=RAM_SIZE, noverlap=PRE_SIZE)
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
    sgr_log = 10 * np.log10(sgr + 1e-20)

    plt.figure(figsize=(9, 3))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()


def output(size, fs):
    samples = list()
    for i in range(size):
        samples.append(i/fs)
    cos1 = np.cos(2 * math.pi * f1 * np.array(samples))
    cos2 = np.cos(2 * math.pi * f2 * np.array(samples))
    cos3 = np.cos(2 * math.pi * f3 * np.array(samples))
    cos4 = np.cos(2 * math.pi * f4 * np.array(samples))

    cos_final = cos1 + cos2 + cos3 + cos4
    spektogram(cos_final, fs)

    # upravenie amplitudy vystupu nech je to nieco pocuvatelne
    m = np.max(np.abs(cos_final))
    cos_final = cos_final/m
    sigi16 = (cos_final * np.iinfo(np.int16).max).astype(np.int16)

    wavfile.write("../audio/4cos.wav", fs, sigi16)


def frekv_char(w, H, fs):
    _, ax = plt.subplots(1, 2, figsize=(8, 3))

    ax[0].plot(w / 2 / np.pi * fs, np.abs(H))
    ax[0].set_xlabel('Frekvence [Hz]')
    ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')

    ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
    ax[1].set_xlabel('Frekvence [Hz]')
    ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

    for ax1 in ax:
        ax1.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()


def imp_od(n_imp, h1):
    plt.figure(figsize=(5, 3))
    plt.stem(np.arange(n_imp), h1, basefmt=' ')
    plt.gca().set_xlabel('$n$')
    plt.gca().set_title('Impulsní odezva $h[n]$')

    plt.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()


def nul_pol(p, z):
    plt.figure(figsize=(4, 3.5))

    # jednotkova kruznice
    ang = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(ang), np.sin(ang))

    # nuly, poly
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

    plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper right')

    plt.tight_layout()


def filtre(f, fs, data):
    norm = fs/2
    n1, wn1 = ss.buttord(wp=[(f-65)/norm, (f+65)/norm], ws=[(f-15)/norm, (f+15)/norm], gstop=40, gpass=3)
    b1, a1 = ss.butter(N=n1, Wn=wn1, btype="bandstop", output="ba")
    print("Filter pre " + str(f) + ":", end=" ")
    print(b1, a1)

    # impulsni odezva
    n_imp = 32
    imp = [1, *np.zeros(n_imp - 1)]  # jednotkovy impuls
    h1 = ss.lfilter(b1, a1, imp)

    # frekvencni charakteristika
    w1, H1 = ss.freqz(b1, a1)

    # nuly, poly
    z1, p1, k1 = ss.tf2zpk(b1, a1)

    print("Nuly a poly: ", end=" ")
    print(z1, p1)

    # stabilita
    is_stable = (p1.size == 0) or np.all(np.abs(p1) < 1)
    print("Is stable:", end=" ")
    print(is_stable)

    # filtrace
    sf1 = ss.lfilter(b1, a1, data)

    imp_od(n_imp, h1)
    frekv_char(w1, H1, fs)
    nul_pol(p1, z1)

    return sf1


def vyfiltuj(data, fs):
    filtered = filtre(f1, fs, data)
    filtered = filtre(f2, fs, filtered)
    filtered = filtre(f3, fs, filtered)
    filtered = filtre(f4, fs, filtered)

    spektogram(filtered, fs)

    average = np.average(filtered)
    # print(average)
    filtered = filtered - average
    filtered = filtered / (abs(filtered).max())

    plot(fs, np.array(filtered))

    sigi16 = (filtered * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write("../audio/clean_bandstop.wav", fs, sigi16)


if __name__ == '__main__':
    freqv, input_signal = take_input()
    ramce = predrobenie(input_signal, freqv)
    dft(ramce[30], freqv)
    spektogram(input_signal, freqv)
    output(input_signal.size, freqv)
    vyfiltuj(input_signal, freqv)
    plt.show()

