import cv2
import numpy as np
import tensorflow as tf
import requests

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="AI_int16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_size = 96
esp32_url = 'http://192.168.42.39/stream'
esp32_s3_ip = 'http://192.168.42.197'  # Replace with your ESP32-S3 IP

cap = cv2.VideoCapture(0)  # Or use esp32_url

prev_state = None  # Keep track of last light state

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_resized = cv2.resize(frame, (img_size, img_size))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(frame_normalized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    label = "human" if prediction > 0.5 else "no human"
    color = (0, 255, 0) if label == "human" else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("ESP32-CAM Stream", frame)

    # Only send command if state changes
    if label != prev_state:
        try:
            if label == "human":
                requests.get(f"{esp32_s3_ip}/light/on")
                print("Light ON command sent")
            else:
                requests.get(f"{esp32_s3_ip}/light/off")
                print("Light OFF command sent")
            prev_state = label
        except Exception as e:
            print(f"Failed to send request: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
