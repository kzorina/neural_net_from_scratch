# A feed-forward multi-layer perceptron from scratch

For training we used car evalusation dataset from:
https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

For comparison we implemented network with recuired architecture in torch (file ```torch_net (for comparison).ipynb```)

History of learning and accuracy 
![alt text](https://github.com/kzorina/neural_net_from_scratch/blob/master/history.png)

Best accuracy with parameters close to recuired ones (task 6)
```params:  0.007 lr, batch size 16 and 13 epochs. 14 1 hid, 34 2 hid ```
Accuracy = 0.896
![alt text](https://github.com/kzorina/neural_net_from_scratch/blob/master/net with batch size 16 and 13 epochs. 14 1 hid, 34 2 hid.png)
