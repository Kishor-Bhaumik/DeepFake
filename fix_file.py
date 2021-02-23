import shutil
import os
def fixdir():
    if not os.path.exists("data/test/fake_real"):
        os.rename("data/test","data/fake_real")
        os.mkdir("data/test")
        shutil.move("data/fake_real", "data/test")
