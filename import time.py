import time

countdown_time = int(input("Enter the countdown time in seconds: "))

while countdown_time:
    mins, secs = divmod(countdown_time, 60)
    timer = '{:02d}:{:02d}'.format(mins, secs)
    print(timer, end="\r")
    time.sleep(1)
    countdown_time -= 1

print("Countdown Complete!")

