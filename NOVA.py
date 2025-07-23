import cv2
import os
import numpy as np
import speech_recognition as sr
import openai
import pyttsx3
import threading
from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

model = YOLO("yolov8n.pt")  

# Configura tu clave API de OpenAI
openai.api_key = "TU_CLAVE_API_DE_OPENAI"

voice_recognizer = sr.Recognizer()
engine = pyttsx3.init()

voices = engine.getProperty('voices')
for voice in voices:
    if 'spanish' in voice.languages:
        engine.setProperty('voice', voice.id)
        break

vision_data = {
    "face_detected": False,
    "person_name": "desconocido",
    "person_name_guardar": "",
    "objects_detected": []  # Lista para almacenar objetos detectados
}
vision_thread = None  
vision_running = False  

def capture_voice():
    with sr.Microphone() as source:
        print("Escuchando...")
        voice_recognizer.adjust_for_ambient_noise(source)
        audio = voice_recognizer.listen(source)
        try:
            command = voice_recognizer.recognize_google(audio, language='es-ES')
            print("Tú dijiste: " + command)
            return command.lower()
        except sr.UnknownValueError:
            print("No entendí lo que dijiste")
            return None
        except sr.RequestError:
            print("Error en el servicio de reconocimiento de voz")
            return None


def analyze_command_with_openai_and_persona(command):
    persona = vision_data['person_name']
    objects_detected = vision_data['objects_detected']
    
    # Crear una descripción del contexto visual
    if objects_detected:
        objects_list = ", ".join([f"{obj['name']} (confianza: {obj['confidence']:.2f})" for obj in objects_detected])
        context_description = f"Estás viendo a {persona} y los siguientes objetos: {objects_list}."
    else:
        context_description = f"Estás viendo a {persona} y no se han detectado objetos específicos en el entorno."


    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": f"Eres NOVA (Neural Organization for Voice and Assistance), un asistente inteligente creado para ayudar a las personas. Respondes de manera amable y concisa. {context_description}"},
                {"role": "user", "content": command}
            ],
            max_tokens=100
        )
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"Error en la solicitud a OpenAI: {e}")
        return None


def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def save_new_face():
    global vision_thread, vision_running
    print('Entrando a la función save_new_face')
    vision_running = False  
    if vision_thread is not None:
        vision_thread.join()  

    dataPath = './data'
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    cap = cv2.VideoCapture(0)
    nameDir = vision_data['person_name_guardar']
    personPath = os.path.join(dataPath, nameDir)

    if not os.path.exists(personPath):
        os.makedirs(personPath)

    print("Capturando imágenes. Presiona 'q' para salir.")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar la imagen.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Captura de imágenes', gray)

        # Detección de caras
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            # Solo procesar la primera cara detectada
            (x, y, w, h) = faces[0]
            filePath = os.path.join(personPath, f'foto_{count}.jpg')
            cv2.imwrite(filePath, gray[y:y+h, x:x+w])  # Guardar solo la cara detectada
            count += 1
            

            if count >= 300:  
                print(f"Se ha capturado la imagen de {nameDir}.")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Llamar a la función de entrenamiento después de la captura
    train_model(dataPath)

    
    start_vision_thread()

def train_model(dataPath):
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)
    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        print('Leyendo las imágenes')
        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            img = cv2.imread(os.path.join(personPath, fileName), 0)
            facesData.append(img)
            cv2.imshow('Leyendo imagen', img)
            cv2.waitKey(100)

        label += 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Entrenando...")
    recognizer.train(facesData, np.array(labels))
    recognizer.write("model.xml")
    print("Modelo entrenado y guardado como model.xml")

def perform_action(command):
    global vision_data
    if "rostro" in command:
        if vision_data["face_detected"]:
                speak_text(f"Veo a {vision_data['person_name']}. Hola, bienvenido.")
    if "nuevo" in command and "rostro" in command:
            speak_text("No te reconozco. ¿Deseas ser reconocido? Dime tu nombre.")
            name_command = capture_voice()
            if name_command:
                vision_data['person_name_guardar']=name_command
                save_new_face()
    elif "abre spotify" in command:
        os.system("start spotify")
        speak_text("Abriendo Spotify")
    elif "abre calculadora" in command:
        os.system("calc")
        speak_text("Abriendo calculadora")
    elif "salir" in command:
        speak_text("Saliendo...")
        exit(0)  # Salir del programa
    elif "abre youtube y pon" in command:
        import webbrowser
        song = command.replace("abre youtube y pon", "").strip()
        search_url = f"https://www.youtube.com/results?search_query={song}"
        webbrowser.open(search_url)
        speak_text(f"Buscando {song} en YouTube.")
    else:
        response = analyze_command_with_openai_and_persona(command)
        if response:
            print("Respuesta del sistema: " + response)
            speak_text(response)



import cv2
import os
from ultralytics import YOLO

vision_data = {
    "face_detected": False,
    "person_name": "desconocido",
    "person_name_guardar": "",
    "objects_detected": []  
}

def vision_agent():
    global vision_data, vision_running
    dataPath = './data' 
    imagePaths = os.listdir(dataPath)
    

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('model.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    yolo_model = YOLO("yolov8n.pt")  

    vision_running = True 
    while vision_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        # Detección de rostros
        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 70:
                vision_data["person_name"] = imagePaths[result[0]]
                vision_data["face_detected"] = True
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                vision_data["person_name"] = 'Desconocido'
                cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Detección de objetos con YOLOv8
        yolo_results = yolo_model(frame)
        annotated_frame = yolo_results[0].plot()  

        # Actualiza la lista de objetos detectados en vision_data
        vision_data["objects_detected"] = []
        for obj in yolo_results[0].boxes:
            class_id = int(obj.cls[0])
            class_name = yolo_results[0].names[class_id]
            confidence = obj.conf[0]
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Coordenadas de la caja de detección
            
            # Añadir el nombre del objeto y la confianza al diccionario
            vision_data["objects_detected"].append({"name": class_name, "confidence": confidence})

            # Dibujar la caja y el nombre del objeto en el cuadro
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Mostrar el cuadro anotado
        cv2.imshow('frame', annotated_frame)

        # Salir con la tecla 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def start_vision_thread():
    global vision_thread, vision_running
    vision_running = True  
    vision_thread = threading.Thread(target=vision_agent)
    vision_thread.start()

start_vision_thread()

while True:
    voice_command = capture_voice()
    if voice_command:
        perform_action(voice_command)