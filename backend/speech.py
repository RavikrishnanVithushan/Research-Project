import pyttsx3

engine = pyttsx3.init()

def speak(word):

    engine.say(word)

    engine.runAndWait()