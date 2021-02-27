# a = 0.353772
# b = 0.324035
# c = (a+b) / 2
# import random
# for _ in range(3):
#     print(c+(a-b)/2*(random.randrange(1000)-500)/500)

# import pandas as pd
# a = [2.378798129,2.379137925,2.379728386,2.378665383,2.380813182,2.379428472,2.378278658,2.37854379,2.37832403,2.377241082,2.378306558,2.378138779]
# b = [0.14646,0.14865,0.145923,0.137563,0.145271,0.144723032,0.142944,0.131826,0.147774,0.137352,0.144257,0.140717264]
#
# data = pd.DataFrame({'a':a,'b':b})
# print(data.corr())

from filelock import Timeout, FileLock

file_path = "high_ground.txt"
lock_path = "high_ground.txt.lock"

lock = FileLock(lock_path, timeout=1)
with lock:
    open(file_path, "a").write("Hello there!")

lock.acquire()
try:
    open(file_path, "a").write("General Kenobi!")
finally:
    lock.release()