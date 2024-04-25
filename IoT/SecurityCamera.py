import random

class SecurityCamera:
    def __init__(self, id, status, security_status, light_ref):
        self.id = id
        self.status = status
        self.security_status = security_status
        self.light_ref = light_ref  # Reference to a SmartLight object

    def toggle(self):
        if self.status == "on":
            self.status = "off"
        else:
            self.status = "on"

    def turn_on(self):
        self.status = "on"

    def turn_off(self):
        self.status = "off"

    def motion_detect(self, isMotion):
        if self.status == "on" and isMotion and self.security_status == "detectMotion":
            # If motion is detected and the camera status is set to detect motion, turn the lights on using the reference
            self.light_ref.turn_on()
            self.light_ref.change_Brightness(100)

    def getStatus(self):
        return self.status


    def getID(self):
        return self.id

    def getSecurity_Status(self):
        return self.security_status

    def changeMode(self):
        if self.security_status == "secure":
            self.security_status = "detectMotion"
        else:
            self.security_status = "secure"

    def randomize(self):
        motion = [True, False]
        return random.choice(motion)