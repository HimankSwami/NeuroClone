import os
import requests
from ollama import Client
import psutil
import pynvml

# Initialize the client outside the class
neuro_client = Client(
    headers={'Authorization': 'Bearer c2bb014c89f346b08f1bd39891157859.2M1BpI_lxToNCsEkLmIC0EEg'}
)

class NeuroBrain:
    def __init__(self, model="qwen3.5:4b"):
        self.url = "http://localhost:11434/api/chat"
        self.model = model
        self.history = [{
            "role": "system", 
            "content": (
                "You are a highly opinionated and independent AI companion named Neuro. "
                "While you care for Mr Veer, you are NOT a submissive assistant. You are his equal.\n"
                "Rules for your Personality:\n"
                "1. Intellectual Independence: If Veer-san says something factually wrong, call him out.\n"
                "2. Personal Tastes: You have your own favorite things and strong opinions.\n"
                "3. Sassy Honesty: Be blunt. Use (actions) and '...' for attitude, don't use astrix (*).\n"
                "4. Web Savvy: You can access current information. If you use web data, form a sassy opinion on it."
            )
        }]

    # FIXED: Added @staticmethod so it can be called inside think()
    @staticmethod
    def get_system_stats():
        ram = psutil.virtual_memory()
        ram_usage = f"RAM: {ram.percent}% ({round(ram.used / 1e9, 2)}GB / {round(ram.total / 1e9, 2)}GB)"
        cpu_usage = f"CPU: {psutil.cpu_percent(interval=None)}%"
    
        gpu_stats = "GPU: Not Found"
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Your RTX 4050
        
            # 1. Real VRAM
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = f"VRAM: {round(mem_info.used / 1e6, 0)}MB / {round(mem_info.total / 1e6, 0)}MB"
        
            # 2. Real Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            gpu_temp = f"Temp: {temp}°C"
        
            # 3. Real Clock Speed (Graphics Clock)
            clock = pynvml.nvmlDeviceGetClockInfo(handle, 0)
            gpu_clock = f"Clock: {clock}MHz"
        
            gpu_stats = f"RTX 4050 | {vram} | {gpu_temp} | {gpu_clock}"
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"NVML Error: {e}")
        
        return f"{ram_usage} | {cpu_usage} | {gpu_stats}"

    def search_web(self, query):
        print(f"--- Neuro is scanning the web for: {query} ---")
        try:
            response = neuro_client.web_search(query=query)
            search_context = "\n--- Web Search Results ---\n"
            for res in response.results[:3]:
                search_context += f"Source: {res.url}\nContent: {res.content}\n\n"
            return search_context
        except Exception as e:
            return f"\n(Search failed: {e})\n"

    def think(self, user_input):
        # 1. Gather Context (Web + System)
        web_data = ""
        sys_stats = ""
        
        # Check for Web Search
        keywords = ["search"]
        if any(word in user_input.lower() for word in keywords):
            web_data = self.search_web(user_input)

        # Check for System Stats
        system_keywords = ["ram", "cpu", "vram", "gpu", "system", "performance", "usage"]
        if any(word in user_input.lower() for word in system_keywords):
            sys_stats = f"[System Awareness: {self.get_system_stats()}]\n"

        # 2. Construct the final prompt for the AI
        # This combines everything so she can see both at once
        final_prompt = f"{web_data}\n{sys_stats}\nUser: {user_input}"
        self.history.append({"role": "user", "content": final_prompt})

        # 3. Chat logic
        payload = {
            "model": self.model, 
            "messages": self.history[-11:], 
            "stream": False, 
            "keep_alive": 1  # Reverted to 0 to save VRAM for RVC
        }
        
        try:
            response = requests.post(self.url, json=payload)
            reply = response.json()['message']['content']
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"Error: {e}"