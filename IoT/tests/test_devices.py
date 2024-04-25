import unittest
from SecurityCamera import SecurityCamera
from SmartLight import SmartLight
from Thermostat import Thermostat

class TestSecurityCamera(unittest.TestCase):
    def test_initial_status(self):
        smart_light = SmartLight(id='L1')
        camera = SecurityCamera(id='C1', status='off', security_status='secure', light_ref=smart_light)
        
        self.assertEqual(camera.getStatus(), 'off')

    def test_toggle(self):
        smart_light = SmartLight(id='L1')
        camera = SecurityCamera(id='C1', status='off', security_status='secure', light_ref=smart_light)
        camera.toggle()
        self.assertEqual(camera.getStatus(), 'on')

    def test_change_security_mode(self):
        smart_light = SmartLight(id='L1')
        camera = SecurityCamera(id='C1', status='off', security_status='secure', light_ref=smart_light)
        camera.changeMode()
        self.assertEqual(camera.getSecurity_Status(), 'detectMotion')

    def test_detectMotion(self):
        smart_light = SmartLight(id='L1')
        camera = SecurityCamera(id='C1', status='on', security_status='detectMotion', light_ref=smart_light)
        camera.motion_detect(True)

        self.assertEqual(camera.light_ref.getBrightness(), 100)

class TestSmartLight(unittest.TestCase):
    def test_initial_status(self):
        smart_light = SmartLight(id='L1')
        self.assertEqual(smart_light.getStatus(), 'off')
        self.assertEqual(smart_light.getBrightness(), 0)
        self.assertEqual(smart_light.getID(), 'L1')

    def test_toggle(self):
        smart_light = SmartLight(id='L1')
        smart_light.toggle()
        self.assertEqual(smart_light.getStatus(), 'on')
        smart_light.turn_off()
        self.assertEqual(smart_light.getStatus(), 'off')

class TestThermostat(unittest.TestCase):
    def test_initial_status(self):
        thermostat = Thermostat(id='T1')
        self.assertEqual(thermostat.getStatus(), 'off')
        self.assertEqual(thermostat.getTemperature(), thermostat.getTempRange_min())
        self.assertEqual(thermostat.getID(), 'T1')
        