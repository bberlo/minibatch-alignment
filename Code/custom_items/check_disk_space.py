import os
import shutil

BytesPerGB = 1024 * 1024 * 1024

(total, used, free) = shutil.disk_usage("/local/")
print("Total: %.2fGB" % (float(total) / BytesPerGB))
print("Used:  %.2fGB" % (float(used) / BytesPerGB))
print("free:  %.2fGB" % (float(free) / BytesPerGB))

print('local:', os.listdir('/local/'))
print('local/20184025:', os.listdir('/local/20184025'))
