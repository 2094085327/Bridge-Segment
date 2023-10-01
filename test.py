import threading 
import time 

def run(): 
	while True: 
		print('thread running') 
		global stop_threads 
		if stop_threads: 
			break

stop_threads = False
t1 = threading.Thread(target = run) 
t1.start() 
stop_threads = True
t1.join() 
print('thread killed111111111111111111111111111')


stop_threads = False
# t1 = threading.Thread(target = run)
t1.start()
stop_threads = True
t1.join()
print('thread killed2')