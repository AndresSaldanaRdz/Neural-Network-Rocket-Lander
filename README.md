# Neural-Network-Rocket-Lander

This repository holds the codes of a neural network I built from scratch. Its purpose is to learn how to control and land a rocket in a computer game. The number of neurons in the input, hidden, and output layers, as well as the learning rate, number of epochs and stopping accuracy  can be easily customized to tune the network. The derived network consisted of 2 input neurons, 8 hidden layer neurons, 2 output layer neurons, a 0.8 learning rate, 9 epochs to reach the stopping criteria, and 50,000 rows in each epoch. When the neural network finishes training, the corresponding weights between each layer are saved to a weights.txt file so that they can be loaded into the game.

Data.csv, data for successful rocket landings taken from the game, it has 4 columns, X and Y distance to target (input) as well as X and Y velocity (output).

Train.ipynb, is where the network can be trained and tuned. Running this file creates a network from scratch, which uses the information from data.csv to learn and produce a weights.txt file that can be loaded into the game.

Game_functions.py, This file holds all the necessary functions needed to execute the NeuralNetHolder File

NeuralNetHolder, the file that instructs the game to use the neural network to control the rocket.

Landing 1 and 2, is just evidence of the code and game working.

To run the game with the neural network (on windows); 

1. Download the Game.zip file (unavailable right now as its too big)
2. Replace "NeuralNetHolder.py" in the downloaded folder and add Game_functions.py.
3. Navigate to the stripes folder and type “. \activate” to launch the virtual environment
4. Go back to the main folder and install the requirements with “pip install -r . \requirements.txt”
5. Launch the program “python . \Main.py”
