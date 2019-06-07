# Final Project

Roy Jara, rjara@ucsd.edu

Elliott Lao, eylao@ucsd.edu

## Abstract Proposal

In this project we intend to go through a more abstract path in generating art by using machine learning or more specifically neutal networks. With video as an input to a convolutional neural network, we generate two outputs, an (x,y) coordinate pair and a MIDI number. The x,y coordinate selects the embedding from a pregenerated embedding of visual content, this could be images or sketches pregenerated with another machine learning model. The x,y coordinate therefore selects the embedding to be the corresponding frame to the input frame of the video. Using ffmpeg we can generate the video frame by frame from the selected embeddings as well as audio generated from the midi notes. The ideal scenario would be to implement this in real time but showing the input video and the output video side by side would also be a nice way to present this project. 

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
- Jupyter notebooks: generative_code.ipynb

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- What you include here will very much depend on the format of your final project
  - image files (`.jpg`, `.png` or whatever else is appropriate)
  - 3d models
  - movie files (uploaded to youtube or vimeo due to github file size limits)
  - audio files
  - ... some other form

## Technical Notes

Any implementation details or notes we need to repeat your work. 
- Does this code require other pip packages, software, etc?
- Does it run on some other (non-datahub) platform? (CoLab, etc.)

## Reference

References to any papers, techniques, repositories you used:
- Papers:
--StyleGAN Paper: http://stylegan.xyz/paper
- Repositories:
  - tf-openpose variation https://github.com/ildoonet/tf-pose-estimation
  - StyleGAN https://github.com/NVlabs/stylegan
- Blog posts
