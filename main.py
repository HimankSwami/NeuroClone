# from brain.core import NeuroBrain
# from voice.speaker import speak

# def run_neuro():
#     neuro = NeuroBrain()
#     print("--- Neuro is online (Type 'exit' to quit) ---")
    
#     while True:
#         user_msg = input("You: ")
#         if user_msg.lower() == "exit":
#             break
            
#         response = neuro.think(user_msg)
#         print(f"Neuro: {response}")
        
#         # This will call your piper TTS script
#         speak(response)

# if __name__ == "__main__":
#     run_neuro()

import speech_recognition as sr
import sys
from pynput import keyboard
from brain.core import NeuroBrain
from voice.speaker import speak

def listen():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 5
    
    with sr.Microphone() as source:
        print("\n--- Listening... ---")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=12)
            print("--- Processing... ---")
            text = recognizer.recognize_google(audio)
            
            print(f"\nCaptured: '{text}'")
            print("Action: [Enter] to Send | [r] to Retry | [ctrl+c] to Exit")
            
            # --- Key Detection Logic ---
            while True:
                user_choice = input(">> ").lower().strip()
                if user_choice == "": # Just pressing Enter
                    return text
                elif user_choice == "r":
                    print("Retrying voice capture...")
                    return "RETRY_SIGNAL"
                else:
                    print("Invalid input. Press Enter to send or 'r' to retry.")
                    
        except sr.UnknownValueError:
            return "RETRY_SIGNAL"
        except Exception as e:
            print(f"Error: {e}")
            return None

def run_neuro():
    brain = NeuroBrain()
    print("--- Neuro is online! ---")

    while True:
        user_input = input(f"User: ")

        # Handle the Retry logic
        if user_input == "RETRY_SIGNAL" or user_input is None:
            continue
            
        if "exit" in user_input.lower():
            break

        # If it passes the check, process as normal
        response = brain.think(user_input)
        print(f"Neuro: {response}")
        speak(response)

if __name__ == "__main__":
    run_neuro()