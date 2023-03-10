# Party Cat Image Filter
This project contains code to train and test a Convolutional Neural Network to identify the facial features of cats.

Currently the model works by training for the x and y coordinates of 9 facial features using a database of 9,000 cat images. It is able to identify these features with 62% accuracy. This accuracy was achieved with only 1/3 of the data given and a relativley low number of epochs, so it is likely that I can try to train overnight and improve these numbers.

My next steps are to apply a party hat based on the positioning of the locaiton of the ear locations. One major setback of the model is that it does not recognize if something is a cat or not to begin with. Another future step could include training the model to be able to first distinguish if the image is a cat/cat-looking animal before desperately searching for cat facial features.

For project requirements and layer calculations please consult the submitted google document.