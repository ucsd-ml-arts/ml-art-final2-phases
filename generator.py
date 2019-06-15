import os
import subprocess
os.chdir("/Users/rjara/tf-pose-estimation")
print(os.getcwd())
import run

os.chdir("/Users/rjara/")
print(os.getcwd())

# Uncomment to breakdown video to separate frames and save them
# subprocess.call(['ffmpeg', '-i', 'Movies/pose_track/og-vid.mov','Movies/pose_track/frame%d0.png'])
for i in range(10):
    image_name = "Movies/pose_track/frame%d0.png" %i
    print(image_name)
    # run(model="mobilenet_thin", resize="432x368", image=)
