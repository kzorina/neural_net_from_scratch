# A feed-forward multi-layer perceptron from scratch

For training we used car evaluation dataset from:
https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

Project structure  
```data/```   contains the data we were assigned
- ```car.data```, ```car.names``` - initial data files
- ```car_evaluation.csv``` - preprocessed data with one dimansional Y
- ```car_evaluation_with_one_hot.csv``` - preprocessed data with one-hot-encoded Y. 
Our neural network works with this structure of input data.

```log_files/```  contains log files 
- ```log_grid_search.txt``` - log of finding bet hyperparameters for our MLP
-  ```activations_comparison.txt``` - log of accuracy comparisons for Softmax and Tanh activation functions (for **task 6.2**)


```plots/``` contains main plots for the project

Scripts:
- ```main.py``` - one run of model with high number of epochs with big number of layers to show the power of our model :)
- ```mpl_with_shortcut.py``` - implementation of MLP as a class
- ```activation_functions.py``` - custom implementation of activation functions Tanh, ReLu and Softmax. Each of them is implemented as a class with static methods ```activation``` and ```derivative```. Softmax has a special derivative for cross entropy which is implemented directly in packward pass if network.  

Jupyter Notebooks:
-  ```data-preprocessing.ipynb``` - preprocessing of original data. Result is saved in .csv
- ```Task 6.2.ipynb``` - accomplishment of **task 6.2** with small analysis of results.
- ```torch_net (for comparison).ipynb``` - inplementation of the given MLP structure using pyTorch for comparison with our MLP from scratch

History of learning and accuracy <br>
<img src="https://github.com/kzorina/neural_net_from_scratch/blob/master/plots/200%20epochs%20with%20batch%20size%2020.png" height="600">

Best accuracy with parameters close to recuired ones (task 6) <br>
```params:  0.007 lr, batch size 16 and 13 epochs. 14 1 hid, 34 2 hid ``` <br>
Accuracy = 0.896 <br>
<img src="https://github.com/kzorina/neural_net_from_scratch/blob/master/plots/net%20with%20batch%20size%2016%20and%2013%20epochs.%2014%201%20hid%2C%2034%202%20hid.png" height="600">

