#Libraries 
import matplotlib.pyplot as plt
import numpy as np
#import pydub
from playsound import playsound
from scipy.io import wavfile
#from pydub import AudioSegment

#pydub.AudioSegment.ffmpeg = "D:/PYTHON VIRTUAL ENVS/ffmpeg-7.0"
def audio_filter(ruta_archivo):
    #Play audio
    playsound(ruta_archivo)
    #Read audio file
    sampFreq, sound = wavfile.read(ruta_archivo)
    # Resto del procesamiento de audio...

    print(sound.dtype, sampFreq)

        # Normalice audio to b between - 1 to 1
    sound = sound / 2.0 ** 15

    # Just one channel
    sound = sound[:, 0]

    # Measure in seconds
    length_in_s  = sound.shape[0] / sampFreq
    print("Audio length in seconds: ", length_in_s)

    # Audio plot
    plt.plot(sound[:], "r")
    plt.xlabel("Sound signal")
    plt.tight_layout
    plt.show()

    # Time vector
    time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s
    plt.plot(time, sound[:], "r")
    plt.xlabel("Time signal [s]")
    plt.tight_layout
    plt.show()

    # Add noise to signal
    yerr = (
        0.005 * np.sin(2 * np.pi * 6000.0 * time)
        + 0.008 * np.sin(2 * np.pi * 8000.0 * time)
        + 0.006 * np.sin(2 * np.pi * 2500.0 * time)
    )
    signal = sound + yerr

    # Zoom
    plt.plot(time[6000:7000], signal[6000:7000])
    plt.ylabel("Time [s]")
    plt.xlabel("Amplitude")
    plt.show()

    # Fourier Transform
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1.0 / sampFreq)
    print("Fourier Transform: ", fft_spectrum)
    fft_spectrum_abs = np.abs(fft_spectrum)

    # Plot FFT
    plt.plot(freq, fft_spectrum_abs)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.show()

    # Filter working on FFT domain
    for i, f in enumerate(freq):
        if f > 5900 and f < 6100:
            fft_spectrum[i] = 0.0
    
    noiseless_signal = np.fft.irfft(fft_spectrum)

    # Guardar audio con ruido en formato WAV
    wavfile.write(r"Noisy_Audio.wav", sampFreq, signal)
    
    # Audio plot
    plt.plot(time, noiseless_signal, "r")
    plt.xlabel("Time signal [s]")
    plt.tight_layout
    plt.show()

    wavfile.write(r"Noisy Audio.wav", sampFreq, signal)
    wavfile.write(r"Clean Audio.wav", sampFreq, noiseless_signal)
    #playsound("Noisy Audio.wav", 'latin1')
    #playsound("Noiseless Audio.wav")

    """
    # Guardar audio con ruido en formato MP3
    noisy_audio = AudioSegment.from_wav("Noisy_Audio.wav")
    noisy_audio.export("Noisy_Audio.mp3", format="mp3")

    # Guardar audio filtrado en formato WAV
    wavfile.write(r"Clean_Audio.wav", sampFreq, noiseless_signal)

    # Guardar audio filtrado en formato MP3
    clean_audio = AudioSegment.from_wav("Clean_Audio.wav")
    clean_audio.export("Clean_Audio.mp3", format="mp3")
    """


if __name__ == "__main__":
    audio_filter()