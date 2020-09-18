# Turtle-Neck-Syndrome-Prevention-App
This is turtle neck symdrome prevention app named "목이 거북해"
This app uses the user's cam to measure the user's neck posture, and connects this to the game to enjoy the correct posture. 
The game part is currently being implemented, and only the part that measures the neck posture is completed.

# Neck posture measurement principle
The haar algorithm was used for recognition, and the posture of the neck was measured by comparing the difference in the relative sizes of the eyes and mouth when moving the neck.

# How to Run code
Two libraries are required to run this program, numpy and opencv.
 - pip install numpy
 - pip install opencv-python == 2.~
 
 
Install numpy and opencv-python using this command. In the case of opencv, you must install the 2.~ version, otherwise it may not work properly.
