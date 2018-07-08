# A feed-forward multi-layer perceptron from scratch

For training we used car evalusation dataset from:
https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

For comparison we implemented network with recuired architecture in torch (file ```torch_net (for comparison).ipynb```)

History of learning and accuracy <br>
<img src="https://github.com/kzorina/neural_net_from_scratch/blob/master/plots/200%20epochs%20with%20batch%20size%2020.png" height="600">

Best accuracy with parameters close to recuired ones (task 6)
```params:  0.007 lr, batch size 16 and 13 epochs. 14 1 hid, 34 2 hid ```
Accuracy = 0.896 <br>
<img src="https://github.com/kzorina/neural_net_from_scratch/blob/master/plots/net%20with%20batch%20size%2016%20and%2013%20epochs.%2014%201%20hid%2C%2034%202%20hid.png" height="600">

