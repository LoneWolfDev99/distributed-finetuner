import os
import json
import time

from datetime import datetime

write_at = "/data/test_retry.json"

arg = os.environ.get("arg")
print(arg)
if arg:
    file = open(write_at, 'w')
    data = {}
    for i in range(10):
        data[i] = datetime.now().strftime("%s")
        time.sleep(1)
    print(data)
    file.write(json.dumps(data))
    file.close()
else:
    file = open(write_at, 'r')
    print(json.load(file))
    timestamp = datetime.now().strftime("%s")
    print(f"stage 2 : {timestamp}" )
    raise Exception("exit_now")
