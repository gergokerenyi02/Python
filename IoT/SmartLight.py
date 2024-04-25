class SmartLight:
    def __init__(self, id):
        self.id = id
        self.status = "off"
        self.brightness = 0

    # Getter methods for fields
    def getStatus(self):
        return self.status

    def getBrightness(self):
        return self.brightness

    def getID(self):
        return self.id


    # Toggle method
    def toggle(self):
        if self.status == "on":
            self.status = "off"
        else:
            self.status = "on"

    # Turn on method
    def turn_on(self):
        self.status = "on"

    # Turn off method
    def turn_off(self):
        self.status = "off"


    # Gradual dimming for light
    def gradual_dim(self, target_brightness, dimmingValue=1):
        if self.status == "on":
            while self.brightness > target_brightness:
                self.brightness -= dimmingValue
                time.sleep(0.1)
            self.brightness = target_brightness

    # Setter
    def change_Brightness(self, newValue):
        self.brightness = int(newValue)
