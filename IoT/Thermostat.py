class Thermostat:
    def __init__(self, id):
        self.id = id
        self.status = "off"
        self.tempRange_min = 0
        self.temperature = self.tempRange_min
        self.tempRange_max = 100


    def toggle(self):
        if self.status == "on":
            self.status = "off"
        else:
            self.status = "on"

    def turn_on(self):
        self.status = "on"

    def turn_off(self):
        self.status = "off"

    def change_Temperature(self, newValue):
        self.temperature = newValue

    def getStatus(self):
        return self.status

    def getTemperature(self):
        return self.temperature

    def getID(self):
        return self.id

    def getTempRange_min(self):
        return self.tempRange_min

    def getTempRange_max(self):
        return self.tempRange_max

    def setRange(self, min, max):
        self.tempRange_min = min
        self.tempRange_max = max