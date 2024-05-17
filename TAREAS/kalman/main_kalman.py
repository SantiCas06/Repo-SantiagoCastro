"""main_kalman.py"""
import cv2
import numpy as np
import torch
from sort import Sort

# Cargar el modelo YOLOv5 preentrenado desde la biblioteca de Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def object_tracking():
    """ Realizar el seguimiento de objetos en un video con YOLOv5 y SORT"""
    # Cargar el video de prueba

    cap = cv2.VideoCapture(r"Repo-SantiagoCastro\kalman\test.mp4")
    if not cap.isOpened():
        print(f"Error: No se puede abrir el archivo de video")
        return
    # Inicializar el rastreador SORT
    tracker = Sort()

    while True:
        """  Bucle principal para realizar el seguimiento de objetos en un video"""
        # Leer un frame del video
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar la detección de objetos con YOLOv5
        results = model(frame)
        results = results.xyxy[0].numpy()
        # Filtrar solo las detecciones de personas (clase 0 en COCO)
        people_detections = results[results[:, 5] == 0]

        detections = []
        for *bbox, conf, cls in people_detections:
            """ Crear una lista de detecciones en el formato [x1, y1, x2, y2, conf]"""
            x1, y1, x2, y2 = map(int, bbox)
            detections.append([x1, y1, x2, y2, conf])
        detections = np.array(detections)

        # Actualizar el rastreador con las nuevas detecciones
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            """ Iterar sobre los objetos rastreados"""
            x1, y1, x2, y2, obj_id = map(int, obj[:5])
            # Dibujar los rectángulos alrededor de los objetos rastreados
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Poner el ID del objeto rastreado en el frame
            cv2.putText(frame, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Mostrar el frame con las anotaciones
        cv2.imshow("view", frame)


        # Salir del bucle si se presiona la tecla Esc
        if cv2.waitKey(1) == 27:
            """ Salir del bucle si se presiona la tecla Esc"""
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    object_tracking()
