# Final Project

Roy Jara, rjara@ucsd.edu

Elliott Lao, eylao@ucsd.edu

## Abstract Proposal

In this project we intend to go through a more abstract path in generating art by using machine learning or more specifically neural networks. With video as an input to a convolutional neural network (an implementation of OpenPose), we generate one output, an (x,y) coordinate pair based on a chosen body part. The x,y coordinate selects the embedding from a pregenerated embedding of visual content, which can be images or sketches pregenerated with another machine learning model. We chose to use StyleGAN to synthesize faces and made a latent space spanning the different combinations of four seeds (actual people). 
The x,y coordinate is mapped to the embedded space, selecting the current frame which will be the selected frame in the output video. Due to processing power limitations, the output video cannot be generated in real time. However, by presenting the input and output videos side by side the connection between body movement and face shift can be seen. 

Regarding the creative goals, we chose to generate video because we thought moving images are always more attractive to the audience. But also because it adds the dimension of time, allowing the input, the user's movements, to move freely around the canvas and change the output however they want - essentially creating a medium of art which is the latent space of the generated embeddings. The user is able of morphing the face however they want and as fast as they want.

This project can be modified to navigate a different latent space of other images, objects, scenes, sketches, etc.

FIRST STEP: Write up a description (in the form of an abstract) of what you will revisit for your final project. This should be one paragraph clearly describing your concept and approach. What are your desired creative goals? How are you expanding on something we covered in the class? How will you present your work next Wednesday in the final project presentations? 

## Project Report

Upload your project report (4 pages) as a pdf with your repository, following this template: [google docs](https://docs.google.com/document/d/133H59WZBmH6MlAgFSskFLMQITeIC5d9b2iuzsOfa4E8/edit?usp=sharing).

## Model/Data

Briefly describe the files that are included with your repository:
- trained models
- training data (or link to training data)

## Code

Your code for generating your project:
- Python: generative_code.py
  - run2.py - We made changes to the run.py script included with tf-openpose to receive a directory as an argument instead of the image name, to run the code for all the images in the given directory.
  - estimator.py - We altered the estimator.py script included with tf-openpose to write the x,y coordinates of the body parts for each image into a .txt file.
  - text_process.py - reads the .txt file with the coordinates for all of the body parts for each frame of the input video, selects the coordinates of only one body part and uses these coordinates to navigate the embeddings of the StyleGAN output. It also iterates over all of the frames to generate the final video with both the input and output.
- Jupyter notebooks: generative_code.ipynb

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- What you include here will very much depend on the format of your final project
  - movie files (uploaded to youtube or vimeo due to github file size limits)

## Technical Notes

Any implementation details or notes we need to repeat your work. 
- Does this code require other pip packages, software, etc?
- Datahub: StyleGAN example, with the faces dataset.
- tf-openpose can be run on a local computer.
- Does it run on some other (non-datahub) platform? (CoLab, etc.)
  - With pregenerated embeddings, the scripts can be run locally on any computer with python and tf-openpose (and it's dependencies)

## Reference

References to any papers, techniques, repositories you used:
- Papers:
--StyleGAN Paper: http://stylegan.xyz/paper
- Repositories:
  - tf-openpose variation https://github.com/ildoonet/tf-pose-estimation
  - StyleGAN https://github.com/NVlabs/stylegan
- Blog posts
