#事前準備
#$ sudo apt-get update
#$ sudo apt-get install pigpio
#$ pip install pigpio
#$ sudo pigpiod
#次回から自動起動のため、↑を/etc/rc.localに記入

PWM_STEERING_PIN = "PIGPIO.BCM.13"           # PWM output pin for steering servo
PWM_THROTTLE_PIN = "PIGPIO.BCM.18"           # PWM output pin for ESC

STEERING_LEFT_PWM = int(4096 * 1 / 20)       # pwm value for full left steering (1ms pulse)
STEERING_RIGHT_PWM = int(4096 * 2 / 20)      # pwm value for full right steering (2ms pulse)

THROTTLE_FORWARD_PWM = int(4096 * 2 / 20)    # pwm value for max forward (2ms pulse)
THROTTLE_STOPPED_PWM = int(4096 * 1.5 / 20)  # pwm value for no movement (1.5ms pulse)
THROTTLE_REVERSE_PWM = int(4096 * 1 / 20)    # pwm value for max reverse throttle (1ms pulse)

#PIGPIO RC control
STEERING_RC_GPIO = 12
THROTTLE_RC_GPIO = 13
DATA_WIPER_RC_GPIO = 19
PIGPIO_STEERING_MID = 1500         # Adjust this value if your car cannot run in a straight line
PIGPIO_MAX_FORWARD = 2000          # Max throttle to go fowrward. The bigger the faster
PIGPIO_STOPPED_PWM = 1500
PIGPIO_MAX_REVERSE = 1000          # Max throttle to go reverse. The smaller the faster
PIGPIO_SHOW_STEERING_VALUE = False
PIGPIO_INVERT = False
PIGPIO_JITTER = 0.025   # threshold below which no signal is reported

frequency = 60 #Hz, 50Hz->20ms for a cycle
pi.set_PWM_frequency(12, frequency)
STEERING_LEFT_PWM = int(4096 * 1 / 20)       # pwm value for full left steering (1ms pulse)
pi.hardware_PWM(gpio_pin0, frequency, STEERING_LEFT_PWM)

right_max = 65000
mid = 80000
left_max = 95000
pi.hardware_PWM(12, 60, 80000)

#PWMduty: 0-1000000 (1M)
PWMduty = 1000000 
dutyratio = 0.1

pi = pigpio.pi()
pi.set_mode(12, pigpio.OUTPUT)
pi.set_PWM_frequency(12, frequency)



# GPIO18: 2Hz、duty比0.5
pi.hardware_PWM(12, 50, 500000)
# GPIO19: 8Hz、duty比0.1
pi.hardware_PWM(gpio_pin1, 8, full*dutyratio)

time.sleep(5)

pi.set_mode(gpio_pin0, pigpio.INPUT)
pi.set_mode(gpio_pin1, pigpio.INPUT)
pi.stop()
