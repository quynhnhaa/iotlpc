import time
import RPi.GPIO as GPIO


class Servo:
    """Lớp điều khiển một động cơ servo."""
    def __init__(self, pin=18, freq=50):
        """
        Khởi tạo servo trên một chân GPIO cụ thể.
        :param pin: Chân GPIO (theo BCM) nối với dây tín hiệu của servo.
        :param freq: Tần số PWM, thường là 50Hz cho servo.
        """
        self.pin = pin
        self.freq = freq
        self.current_angle = -1  # Trạng thái góc không xác định ban đầu

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, self.freq)
        self.pwm.start(0)

    def set_angle(self, angle):
        """Đặt góc quay cho servo (0-180 độ)."""
        if angle == self.current_angle:
            return  
        duty_cycle = (angle / 18.0) + 2
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  
        self.pwm.ChangeDutyCycle(0) 
        self.current_angle = angle

    def cleanup(self):
        """Dọn dẹp tài nguyên GPIO cho servo này."""
        if self.pwm:
            self.pwm.stop()
        GPIO.cleanup(self.pin)
