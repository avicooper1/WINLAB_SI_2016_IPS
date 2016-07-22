import threading
import time

exitFlag = 0

class MyThread (threading.Thread):
    def __init__(self, toRun, frame, contours, counter):
        threading.Thread.__init__(self)
        self.toRun = toRun
        self.frame = frame
        self.contours = contours
        self.counter = counter
    def run(self):
        self.toRun(self.frame, self.contours, self.counter)

# Create new threads
#thread1 = myThread(1, "Thread-1", 1)
#thread2 = myThread(2, "Thread-2", 2)

# Start new Threads
#thread1.start()
#thread2.start()