import threading
import tkinter as tk
from tkinter import ttk

from SmartLight import SmartLight
from Thermostat import Thermostat
from SecurityCamera import SecurityCamera
from AutomationSystem import AutomationSystem

import random

class IoTSimulatorApp:
    def __init__(self, root):    
        
        self.root = root
        self.root.title("Smart Home IoT Simulator")
        

        # Creating devices
        self.smart_light = SmartLight(id='L1')
        self.thermostat = Thermostat(id='T1')
        self.security_camera = SecurityCamera(id='C1', status='off', security_status='secure', light_ref=self.smart_light)

        

        # DISPLAY
        
        # Creating console log
        self.console = tk.Text(root, height=10, width=50, state="disabled")
        self.console.pack()

        
        # BUTTONS
        self.toggle_light_button = tk.Button(root, text="Toggle Light", command=self.toggle_light)
        self.toggle_thermostat_button = tk.Button(root, text="Toggle Thermostat", command=self.toggle_thermostat)
        self.toggle_camera_button = tk.Button(root, text="Toggle Camera", command=self.toggle_camera)
        self.detect_motion_button = tk.Button(root, text="Detect Random Motion", command=self.detect_random_motion)
        self.cameraMode_button = tk.Button(root, text="Toggle Mode", command=self.toggle_cameraMode)

 

        # SCALES
        self.brightness_scale = ttk.Scale(root, from_=0, to=100, orient="horizontal", command=self.change_light_brightness)
        self.temperature_scale = ttk.Scale(root, from_=self.thermostat.getTempRange_min(), to=self.thermostat.getTempRange_max(), orient="horizontal", command=self.change_temperature)

        # LABELS
        self.light_status_label = tk.Label(root,text=f"Light Status: {self.smart_light.getStatus()}")
        self.thermostat_status_label = tk.Label(root, text=f"Thermostat Status: {self.thermostat.getStatus()}")
        self.camera_status_label = tk.Label(root, text=f"Camera Status: {self.security_camera.getStatus()}")

        # Setting them down into the GUI
        self.console.pack()
        
        self.light_status_label.pack()
        self.toggle_light_button.pack()
        self.brightness_scale.pack()

        self.thermostat_status_label.pack()
        self.toggle_thermostat_button.pack()
        self.temperature_scale.pack()
        
        
        self.camera_status_label.pack()
        self.toggle_camera_button.pack()
        self.detect_motion_button.pack()
        self.cameraMode_button.pack()

        
        
        self.system = AutomationSystem(self.light_status_label, self.thermostat_status_label, self.camera_status_label, self.brightness_scale, self.temperature_scale)
        self.system.add_device(self.smart_light)
        self.system.add_device(self.thermostat)
        self.system.add_device(self.security_camera)

        self.start_simulation()
        
        
        

    def log_message(self, message):
        self.console.config(state="normal")
        self.console.insert("end", message + "\n")
        self.console.see("end") # Always show the end of the console!
        self.console.config(state="disabled")

    def toggle_light(self):
        self.smart_light.toggle()
        self.log_message(f"Light Status: {self.smart_light.getStatus()}")
        self.light_status_label.config(text=f"Light Status: {self.smart_light.getStatus()}")
        

    def toggle_thermostat(self):
        self.thermostat.toggle()
        self.log_message(f"Thermostat Status: {self.thermostat.getStatus()}")
        self.thermostat_status_label.config(text=f"Thermostat Status: {self.thermostat.getStatus()}")
        

    def toggle_camera(self):
        self.security_camera.toggle()
        self.camera_status_label.config(text=f"Camera Status: {self.security_camera.getStatus()}")
        self.log_message(f"Camera Status: {self.security_camera.getStatus()}, Mode: {self.security_camera.getSecurity_Status()}")
        #self.light_status_label.config(text=f"Light Status: {self.smart_light.getStatus()}")
        

    def toggle_cameraMode(self):
        self.security_camera.changeMode()
        self.log_message(f"Mode changed to: {self.security_camera.getSecurity_Status()}")
        
    
    def detect_random_motion(self):
        if self.security_camera.getStatus() == "on" and self.security_camera.getSecurity_Status() == "detectMotion":
            motion_detected = random.choice([True, False])
            if motion_detected:
                self.security_camera.motion_detect(True)
                self.log_message("Motion detected at the door.")
                self.brightness_scale.config(value=100)
                self.light_status_label.config(text=f"Light ID: {self.smart_light.getID()}, State: {self.smart_light.getStatus()}, Brightness: {self.smart_light.getBrightness()}")
                
            else:
                self.log_message("No motion detected.")
        elif self.security_camera.getStatus() == "on" and self.security_camera.getSecurity_Status() == "secure":
            self.log_message("Camera mode is currently on secure. Turn on motion detection to check motion.")
        else:
            self.log_message("Camera is off. To detect motion, turn on the camera!")

        #self.light_status_label.config(text=f"Light Status: {self.smart_light.getStatus()}")

    def change_light_brightness(self, event):
        new_brightness = int(self.brightness_scale.get())
        self.smart_light.change_Brightness(new_brightness)
        self.light_status_label.config(text=f"Light ID: {self.smart_light.getID()}, State: {self.smart_light.getStatus()}, Brightness: {self.smart_light.getBrightness()}")
        self.log_message(f"Light Brightness: {self.smart_light.getBrightness()}")

    def change_temperature(self, event):
        new_temperature = int(self.temperature_scale.get())
        self.thermostat.change_Temperature(new_temperature)
        self.log_message(f"Thermostat Temperature: {self.thermostat.getTemperature()}")


    def start_simulation(self):
        
        duration = 3600
        interval = 1

        # Create and start a separate thread for the simulation
        self.simulation_thread = threading.Thread(target=self.system.run_simulation, args=(duration, interval))
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def stop_simulation(self):
        self.is_running = False



if __name__ == "__main__":
    root = tk.Tk()
    app = IoTSimulatorApp(root)

    root.mainloop()