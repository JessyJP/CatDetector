import subprocess

# Variables
scriptName = './cat_detector.py';
outputDir = "./output/"
video = "./catvideo.mp4";

# Define the command to be executed
command = []
command.append(['python', scriptName , video , "-s2", outputDir])
command.append(['python', "cat_detector_withbox.py" , video , "-s2", outputDir])

# Execute the command
for cmd in command:
    subprocess.run(cmd)

