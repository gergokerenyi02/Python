import time
from SmartLight import SmartLight
from Thermostat import Thermostat
from SecurityCamera import SecurityCamera


class AutomationSystem:
    def __init__(self, lightLabel, thermostatLabel, cameraLabel, brightnessValueLabel, temperatureScale):
        self.devices = []  # List to store all devices
        self.lightLabel = lightLabel
        self.thermostatLabel = thermostatLabel
        self.cameraLabel = cameraLabel
        self.brightnessValueLabel = brightnessValueLabel
        self.temperatureScale = temperatureScale
        self.cameraMotion = False 

        # Variables above could be rewritten, since I store the app now and have access for the labels.
        #self.app = app

    def add_device(self, device):
        if self.discover(device):
            self.devices.append(device)
        else:
            # Exception
            return

    def discover(self, device):
        if isinstance(device, SmartLight) or isinstance(device, Thermostat) or isinstance(device, SecurityCamera):
            return True
        else:
            return False

    def run_simulation(self, duration, interval):
        end_time = time.time() + duration
        
        # Check the device states
        while time.time() < end_time:

            # Generate random motion for the camera
            for device in self.devices:
                if isinstance(device, SecurityCamera):
                    self.cameraMotion = device.randomize()

                    time.sleep(10)
                    

                    device.motion_detect(self.cameraMotion)
                    try:
                        self.lightLabel.config(text=f"Light ID: {device.light_ref.getID()}, State: {device.light_ref.getStatus()}, Brightness: {device.light_ref.getBrightness()}")
                        self.brightnessValueLabel.config(value=device.light_ref.getBrightness())
                    except:
                        pass
                   
            time.sleep(interval)