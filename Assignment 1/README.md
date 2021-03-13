# Assignment 1

This directory contains solution to [assignment 1](https://wandb.ai/miteshk/assignments/reports/Assignment-1--VmlldzozNjk4NDE?accessToken=r7ndsh8lf4wlxyjln7phvvfb8ftvc0n4lyn4tiowdg06hhzpzfzki4jrm28wqh44) of Fundamental Deep Learning (CS6910), Spring 2021. It also contains the implementation of a Feed Forward Neural Network **FFNClassifier** which can be trained to work on numerical data.

The solution report with results can be found [here](https://wandb.ai/0x2e4/cs6910-a1/reports/CS6910-Spring-2021-Assignment-1--Vmlldzo1MjA1NjE).

## Usage
The main content of this file is the class FFNClassifier which implements all the required algorithms for the assignment. It can be passed with all the hyperparameters required. The class takes in attributes as follows:
```python
FNNClassifier(self,
                 layer_size,
                 num_layers,
                 activation = 'ReLU',
                 optimizer = 'adam',
                 weight_decay = 0.0001,
                 batch_size = 200,
                 learning_rate = 0.001,
                 num_epochs = 200,
                 weight_init = 'Xavier',
                 loss = 'cross_entropy')
```

**activation : {'sigmoid', 'tanh', 'ReLU'}, default = 'ReLU'**  
The activation function for the hidden layers  
**optimizer : {'normal', 'sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'}, default = 'adam'**  
The optimization algorithm to use  
**weight_decay : float, default = 0.0001**  
L2 regularization hyperparameter  
**batch_size : int, default = 200**  
Batch size for SGD  
**learning_rate : float, default = 0.001**  
The learning rate  
**num_epochs : int, default = 200**  
The number of gradient descent epochs  
**weight_init : {'random', 'Xavier'}, default = 'Xavier'**  
The method of weight initialization   
**loss = {'cross_entropy', 'square'}, default = 'cross_entropy'**  
The loss function to be minimized  

One can fit the training data using the fit function:
```python
model = FFNClassifier(3, 7, optimizer = 'adam', weight_init = 'Xavier')
model.fit(X_train, Y_train)
```

Finally, one can predict using the model, given some test data. To calculate the accuracy using this predicted data, use the accuracy function.
```python
Y_pred = model.predict(X_test)
print(model.accuracy(Y_test, Y_pred))
```

If one wants the predicted probabilities instead of the class, predict_proba can be used instead of predict. To calculate the loss using this, one can call the loss_calc function.
```python
Y_pred_proba = model.predict_proba(X_test)
print(model.loss_calc(Y_test, Y_pred_proba))
```
