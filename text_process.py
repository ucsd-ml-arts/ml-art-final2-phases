from numpy import interp
import numpy as np
import subprocess
import cv2
import os

import re
def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

tracked_parts = ['part #7']  #part 4 is Right Wrist
output_dim = (19,19) #x,y
out_x, out_y = output_dim

x_coords = []
y_coords = []

with open('testfile.txt', "r") as file: #, open('newfile.txt', 'w') as newfile:
    counter=0
    for line in file:
        if any(tracked_part in line for tracked_part in tracked_parts):
            counter = counter+1

            # First number is X coordinate, second is Y coordinate
            coords = line[line.find("(")+1:line.find(")")]
            # print(line[line.find("(")+1:line.find(")")])
            x, y = [int(i) for i in coords.split(", ")]
            x_coords.append(x)
            y_coords.append(y)
            # print("x: %d, y: %d" %(x,y))

print("max x: %d, max y: %d" %(max(x_coords), max(y_coords)))
print("min x: %d, min y: %d" %(min(x_coords), min(y_coords)))

new_xcoords = interp(x_coords, [min(x_coords), max(x_coords)], [0,out_x-1])
new_ycoords = interp(y_coords, [min(y_coords), max(y_coords)], [0,out_y-1])
print("\n")
print("new_max x: %d, new_max y: %d" %(max(new_xcoords), max(new_xcoords)))
print("new_min x: %d, new_min y: %d" %(min(new_ycoords), min(new_ycoords)))

#------------Now make the video:-----------------------------

input_folder1 = 'generated_imgs' #path with the photos generated from StyleGAN
input_folder2 = "real_imgs"
filelist = sorted_nicely(os.listdir(input_folder2))

video_name = 'output_video.avi'

frame = cv2.imread(os.path.join(input_folder1, "%d-%d.png" %(int(new_xcoords[0]), int(new_ycoords[0]))))
frame2 = cv2.imread(os.path.join(input_folder2, "frame1.png"))
height, width, layers = frame.shape
height2, width2, layers2 = frame2.shape

video = cv2.VideoWriter(video_name, 0, 30, (width+width2,height))

with open("processed_frameNames.txt", "w") as file:
    counter = 0
    for x, y in zip(new_xcoords, new_ycoords):

        # files are named by "(row)-(column).png"
        print("new_x: %d, new_y: %d" %(x,y))
        face_img = "%d-%d.png" %(int(x),int(y))
        file.write(face_img+"\n")

        image1 = cv2.imread(os.path.join(input_folder1, face_img))
        image2 = cv2.imread(os.path.join(input_folder2, filelist[counter]))

        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

        #combine 2 images
        vis[:h1, :w1,:3] = image1
        vis[:h2, w1:w1+w2,:3] = image2

        video.write(vis)
        counter = counter + 1

cv2.destroyAllWindows()
video.release()

#-------Now compress video

# subprocess.call(['ffmpeg', '-i', video_name,'compressed_output_video.mov'])
# ffmpeg -i infile.avi youroutput.mp4
