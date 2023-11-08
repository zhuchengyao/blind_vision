import cv2
import time
from ultralytics import YOLO
import os
import openai
import keyboard
import pyttsx3
import get_api

ChatAPI = get_api.get_openai_api()
openai.api_key = ChatAPI
camera = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')
gpt_model = "gpt-3.5-turbo"
pp = pyttsx3.init()
pp.setProperty('rate', 85)

messages = []
# detected_pos = []
# detected_objs_names = []
num_detected_objs = 0




messages.append({"role": "user",
                 "content": "I will send you some data. The first number is the objective that a camera detected. "
                            "first data is how many objectives in front of him. The following is detected "
                            "objectives' name and positions in format of name, position x, position y "
                            "position x and position y are ranged from 0 to 1. For x, close to 0 means on my left side "
                            "close to 1 means the objective is closer to the right side "
                            # "ignore the y variable."
                            "position y is a vertical scale, if the number is lower than 0.5, it means the objective "
                            "is close to you, vice verse."
                            "I need you to format a sentence to notice someone who cannot see. Tell him the objectives"
                            " in front of him. please describe objectives one by one, tell me the "
                            "objective is on which side relative to me, don't tell me the x or y coordinates. "
                            "give me the sentences directly, don't say anything else."})
response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        temperature=0.5,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
messages.append(response.choices[0].message)
print(response.choices[0].message)

while True:
    detected_pos = []
    detected_objs_names = []
    action = input("press any key to 'see', press 'q' to quit")
    if action == 'q':
        break
    ret, frame = camera.read()
    # cv2.imshow('cam', frame)
    results = model(frame)
    img = results[0].plot()
    cv2.imshow('cam', img)
    cv2.waitKey()
    boxes = results[0].boxes
    # the number of detected objects stored in num_detected_objs, use tensor.
    num_detected_objs = len(boxes.xywhn)
    # define OpenAI message.
    # use detected_pos to store the positions of detected objectives.
    for xywhn in boxes.xywhn:
        x = xywhn[0].item()
        x = round(x, 3)
        detected_pos.append(x)
        y = xywhn[1].item()
        y = round(y, 3)
        detected_pos.append(y)

    # use detected_objs_names to store names of objectives
    name_list = boxes.cls.tolist()

    for name_id in name_list:
        detected_objs_names.append(results[0].names[int(name_id)])

    print(detected_objs_names)
    print(detected_pos)
    print(num_detected_objs)
    # now we have the variables we need.
    # num_detected_objs is the number of detected objectives,
    # detected_pos are the positions of these detected objectives,
    # detected_objs_names are the names of these detected objectives.
    message = 'number of objectives is ' + (str(num_detected_objs))
    for i in range(num_detected_objs):
        message = message + 'the number' + str(i+1) + 'objective is ' + str(detected_objs_names[i]) \
                  + 'its x position is ' + str(float(detected_pos[2*i])) + 'its y position is ' + str(float(detected_pos[2*i+1]))+'. '
    messages.append({"role": "user",
                     "content": message})
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text = response.choices[0].message["content"]
    print(text)
    pp.say(text)
    pp.runAndWait()
    cv2.destroyAllWindows()
