import cv2
import tkinter as tk
from PIL import Image,ImageTk

from backend.predict import predict_sign
from backend.world_builder import add_letter,clear_word
from backend.speech import speak

cap = cv2.VideoCapture(0)

def capture_letter():

    ret,frame = cap.read()

    roi = frame[100:400,100:400]

    letter = predict_sign(roi)

    word = add_letter(letter)

    label_word.config(text=word)


def speak_word():

    speak(label_word["text"])


def clear():

    word = clear_word()

    label_word.config(text=word)


def update_frame():

    ret,frame = cap.read()

    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame)

    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10,update_frame)


root = tk.Tk()
root.title("Sign Language Recognition System")

video_label = tk.Label(root)
video_label.pack()

label_word = tk.Label(root,font=("Arial",30))
label_word.pack()

btn1 = tk.Button(root,text="Add Letter",command=capture_letter)
btn1.pack()

btn2 = tk.Button(root,text="Speak",command=speak_word)
btn2.pack()

btn3 = tk.Button(root,text="Clear",command=clear)
btn3.pack()

update_frame()

root.mainloop()