# Assisted correction
![Demonstration](https://sebastiengrd.github.io/images_for_other_projects/Screenshot.png)


## Introduction

This project started in summer 2018, when I had 17 y/o. I designed a computer program to automatically score the points that a teacher writes with a red pencil on a student’s exam copy. The goal of this program is to reduce errors related to human distraction when marking exams, a very real phenomenon. Thus, the method will allow teachers to correct long questions, without worrying about counting the marks afterwards. In order to complete this task, it was necessary to use several computer vision and artificial intelligence techniques in order to design a reliable solution, while adapting to different light environments and different writings.

## Project Details

### Librairy used
 - Opencv
 - Tensorflow (keras)
 - Kivy
 - Scypi
 - Joblib
 - Sklearn
 - Imutils

Also used the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

### Artificial intelligence used
 - SVC model for detecting the red color color (trained with a dataset of 200 images that I made myself)
 - Convolutional neural network for classifying each hand-written digits (trained with the MNIST dataset)

## Results
With a Convolutional neural network that has an accuracy of 99.67% and a SVC model that has a very good accuracy (did not test yet with a large enough test set to draw conclusion), this computer program works really well. The computer vision part is also very robust against different light environments.


### Acknowledgments

I would like to thank the [Réseau Technoscience](https://technoscience.ca/) and the [Canada-Wide Science Fair](https://cwsf.youthscience.ca/) for letting me present my project at various science competitions. This is what gave me the motivation to continue to improve my project based on suggestions from the judges.

I would also like to thank the Génial emission at the Télé-Québec channel, one of the most popular science TV shows in Quebec, for letting me presenting my project in one of their episodes. It was a big opportunity for me.


