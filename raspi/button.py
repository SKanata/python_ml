import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.IN)

try:
    while True:
        if GPIO.input(14) == GPIO.HIGH:
            print(1)
        else:
            print(0)
        sleep(0.5)
            
except KeyboardInterrupt:
    pass
    
