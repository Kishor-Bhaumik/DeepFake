from pydrive.drive import GoogleDrive 
from pydrive.auth import GoogleAuth 

# For using listdir() 
import os 


# Below code does the authentication 
# part of the code 
gauth = GoogleAuth() 

# Creates local webserver and auto 
# handles authentication. 
#gauth.LocalWebserverAuth()	
gauth.CommandLineAuth()
drive = GoogleDrive(gauth) 

# replace the value of this variable 
# with the absolute path of the directory 

path = r"model"

# iterating thought all the files/folder 
# of the desired directory 
for x in os.listdir(path):
	f = drive.CreateFile({'title': x}) 
	f.SetContentFile(os.path.join(path, x)) 
	f.Upload() 
 
	f = None



# f = drive.CreateFile({'title': x}) 
# f.SetContentFile(os.path.join(path, x)) 
# f.Upload() 

# f = None
