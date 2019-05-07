import time
import datetime

t = time.time()
print(t)
print(int(t))
print(int(round(t*1000)))

t2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(t2)

t3 = 1557106611
print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(t3)))