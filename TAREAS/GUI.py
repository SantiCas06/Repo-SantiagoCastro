"""Module for filtering audio signals using a GUI."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from pydub import AudioSegment
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    """Funciones de filtro pasa bajas."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Funciones de filtro pasa bajas."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Funciones de filtro pasa altas
def butter_highpass(cutoff, fs, order=5):
    """Funciones de filtro pasa altas."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """Funciones de filtro pasa altas."""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Funciones de filtro pasa banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Funciones de filtro pasa banda."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Funciones de filtro pasa banda."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class MiVentana(QWidget):
    """Clase para la interfaz gráfica de la aplicación."""

    def __init__(self):
        """Clase para la interfaz gráfica de la aplicación."""
        super().__init__()
        self.setWindowTitle("Análisis de Audio")
        self.setGeometry(100, 100, 800, 600)

        # Layout principal
        layout_principal = QVBoxLayout()

        # Etiqueta para mostrar el archivo cargado
        self.label_archivo = QLabel("Archivo de audio: ")
        layout_principal.addWidget(self.label_archivo)

        # Botón para cargar archivo
        self.button_cargar = QPushButton("Cargar archivo")
        self.button_cargar.clicked.connect(self.cargar_archivo)
        layout_principal.addWidget(self.button_cargar)

        # Layout para los controles de procesamiento
        layout_procesamiento = QHBoxLayout()

        # Selector de tipo de filtro
        self.combo_filtro = QComboBox()
        self.combo_filtro.addItems(["Pasa bajas", "Pasa altas", "Pasa banda"])
        layout_procesamiento.addWidget(self.combo_filtro)

        # Etiqueta para mostrar el valor mínimo del slider
        self.label_min = QLabel("0")
        layout_procesamiento.addWidget(self.label_min)

        # Control deslizante para la frecuencia de corte
        self.slider_frecuencia = QSlider(Qt.Horizontal)
        self.slider_frecuencia.setRange(
            0, 100
        )  # Ajuste del rango del slider para el filtro pasa banda
        layout_procesamiento.addWidget(self.slider_frecuencia)

        # Etiqueta para mostrar el valor máximo del slider
        self.label_max = QLabel("100")
        layout_procesamiento.addWidget(self.label_max)

        # Botón para aplicar filtro
        self.button_filtro = QPushButton("Aplicar filtro")
        self.button_filtro.clicked.connect(self.aplicar_filtro)
        layout_procesamiento.addWidget(self.button_filtro)

        layout_principal.addLayout(layout_procesamiento)

        # Botón para aplicar transformada
        self.button_transformada = QPushButton("Aplicar Transformada")
        self.button_transformada.clicked.connect(self.aplicar_transformada)
        layout_principal.addWidget(self.button_transformada)

        # Visualización de la señal de audio
        self.figura, self.ax = plt.subplots()
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Amplitud")
        self.ax.set_title("Señal de audio")
        self.canvas = FigureCanvas(self.figura)
        layout_principal.addWidget(self.canvas)

        # Variables para almacenar los datos del audio
        self.sampFreq = None
        self.sound = None

        self.setLayout(layout_principal)

    def cargar_archivo(self):
        """Método para cargar un archivo de audio."""
        # Método para cargar un archivo de audio
        archivo, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Archivo de Audio",
            "",
            "Archivos de Audio (*.mp3 *.wav *.aac)",
        )
        if archivo:
            self.label_archivo.setText(f"Archivo de audio: {archivo}")

            # Verificar si el archivo es MP3 o AAC y convertirlo a WAV si es necesario
            if archivo.endswith(".mp3"):
                audio = AudioSegment.from_mp3(archivo)
                archivo_wav = archivo[:-4] + ".wav"  # Cambiar la extensión a .wav
                audio.export(archivo_wav, format="wav")
                self.sampFreq, self.sound = wavfile.read(archivo_wav)
            elif archivo.endswith(".aac"):
                audio = AudioSegment.from_file(archivo, format="aac")
                archivo_wav = archivo[:-4] + ".wav"  # Cambiar la extensión a .wav
                audio.export(archivo_wav, format="wav")
                self.sampFreq, self.sound = wavfile.read(archivo_wav)
            else:
                self.sampFreq, self.sound = wavfile.read(archivo)

            self.dibujar_señal()

    def dibujar_señal(self):
        """Método para graficar la señal de audio cargada."""
        # Método para graficar la señal de audio cargada
        if self.sound is not None:
            tiempo = np.arange(self.sound.shape[0]) / self.sampFreq
            self.ax.clear()
            self.ax.plot(tiempo, self.sound)
            self.canvas.draw()

    def aplicar_filtro(self):
        """Función para aplicar un filtro al audio."""
        # Método para aplicar un filtro al audio
        tipo_filtro = (
            self.combo_filtro.currentText()
        )  # Obtener el tipo de filtro seleccionado

        # Obtener la frecuencia de corte del control deslizante y convertirla al rango adecuado
        frecuencia_corte = self.slider_frecuencia.value() / 100

        # Llamar a la función de filtrado con los parámetros seleccionados
        audio_filtrado = audio_filter(
            self.sound, tipo_filtro, frecuencia_corte, self.sampFreq
        )

        # Actualizar la señal en el gráfico
        tiempo = np.arange(audio_filtrado.shape[0]) / self.sampFreq
        self.ax.clear()
        self.ax.plot(tiempo, audio_filtrado)
        self.canvas.draw()

    def aplicar_transformada(self):
        """Método para aplicar la transformada de Fourier al audio."""
        # Método para aplicar la transformada de Fourier al audio
        if self.sound is not None:
            # Calcular la transformada de Fourier
            transformada = np.fft.fft(self.sound)
            magnitud = np.abs(transformada)
            frecuencia = np.fft.fftfreq(len(self.sound), d=1 / self.sampFreq)

            # Limpiar el eje antes de graficar la transformada
            self.ax.clear()
            self.ax.plot(
                frecuencia[: len(frecuencia) // 2], magnitud[: len(magnitud) // 2]
            )
            self.ax.set_xlabel("Frecuencia (Hz)")
            self.ax.set_ylabel("Magnitud")
            self.ax.set_title("Transformada de Fourier")

            # Agregar un label a la gráfica
            self.ax.text(
                0.05,
                1,
                "Señal Filtrada",
                fontsize=12,
                color="red",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
            )
            self.ax.text(
                0.05,
                0.9,
                "Señal Original",
                fontsize=12,
                color="blue",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
            )

            # Redibujar la gráfica
            self.canvas.draw()


# Función para aplicar el filtro seleccionado al audio
def audio_filter(sound, tipo_filtro, frecuencia_corte, sampFreq):
    """Función para aplicar un filtro al audio."""
    # Normalizar audio entre -1 y 1
    sound = sound / 2.0**15

    # Seleccionar un solo canal
    sound = sound[:, 0]

    # Aplicar filtro según el tipo seleccionado
    if tipo_filtro == "Pasa bajas":
        sound_filtrada = butter_lowpass_filter(sound, frecuencia_corte, sampFreq)
    elif tipo_filtro == "Pasa altas":
        sound_filtrada = butter_highpass_filter(sound, frecuencia_corte, sampFreq)
    elif tipo_filtro == "Pasa banda":
        # Definir un rango de frecuencia para el filtro pasa banda
        # El rango se define en Hz y debe ir de 20 Hz a 10000 Hz
        frecuencia_min = 20
        frecuencia_max = 10000
        # Escalar la frecuencia de corte al rango definido
        frecuencia_corte = (
            frecuencia_min + (frecuencia_max - frecuencia_min) * frecuencia_corte
        )
        sound_filtrada = butter_bandpass_filter(
            sound, frecuencia_min, frecuencia_corte, sampFreq
        )

    return sound_filtrada


# Función principal para ejecutar la aplicación
def main():
    """Función principal para ejecutar la aplicación."""
    app = QApplication(sys.argv)
    ventana = MiVentana()
    ventana.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
