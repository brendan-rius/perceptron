# Perceptron

Very simple single unit perceptron implementation with perceptron rule as training algorithm.

```python
# We generate training data for OR perceptron
train_X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
train_Y = np.array([
    0,
    1,
    1,
    1,
])

p = Perceptron(n_inputs=2, learning_rate=0.1)
p.fit(train_X, train_Y)
print("Weights: {}\nThreshold: {}".format(p.weights, p.threshold))
print(p.predict(train_X))
```

gives

```
Weights: [ 0.1  0.1]
Threshold: 0.1
[0, 1, 1, 1]
```