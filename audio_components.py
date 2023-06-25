import speech_recognition as sr
import pyttsx3

def listen_for_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Setting myself up for success...")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Talk to me!")
        while True:
            try:
                audio = recognizer.listen(source, timeout=20, phrase_time_limit=10)
                print("Enscribing your words...")
                text = recognizer.recognize_whisper(audio, model="medium.en")
            except Exception as e:
                unrecognized = f"Whoopsie, I think I misheard you. Exception: {e}"
                print(unrecognized)
            
            if text:
                return text

def speak_your_mind(input):
    engine = pyttsx3.init()
    engine.say(input)
    engine.runAndWait()