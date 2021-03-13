# Assignment 1

This directory contains solution to [assignment 1](https://wandb.ai/miteshk/assignments/reports/Assignment-1--VmlldzozNjk4NDE?accessToken=r7ndsh8lf4wlxyjln7phvvfb8ftvc0n4lyn4tiowdg06hhzpzfzki4jrm28wqh44) of Fundamental Deep Learning (CS6910), Spring 2021. It also contains the implementation of a Feed Forward Neural Network **FFNClassifier** which can be trained to work on numerical data.

The solution report with results can be found [here](https://wandb.ai/0x2e4/cs6910-a1/reports/CS6910-Spring-2021-Assignment-1--Vmlldzo1MjA1NjE).

## Usage
The main content of this file is the class FFNClassifier which implements all the required algorithms for the assignment. It can be passed with all the hyperparameters required. An example follows:

```python
model = FFNClassifier(3, 7, optimizer = 'adam', weight_init = 'Xavier')
```

Next, one can fit the training data using the fit function:
```python
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
print(model.accuracy(Y_test, Y_pred_proba))
```
