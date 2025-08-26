# Mathematical Derivation of Logistic Regression Gradients

## Problem Setup

We have:
- **Input**: X (m × n matrix, where m = samples, n = features)
- **Weights**: W (n × 1 vector)
- **Bias**: b (scalar)
- **True labels**: y (m × 1 vector, values 0 or 1)

## Forward Pass Equations

1. **Linear combination**: `z = X·W + b`
2. **Sigmoid activation**: `a = σ(z) = 1/(1 + e^(-z))`
3. **Cost function**: `J = -1/m × Σ[y·log(a) + (1-y)·log(1-a)]`

## Step 1: Derivative of Sigmoid Function

First, let's find the derivative of the sigmoid function:

```
σ(z) = 1/(1 + e^(-z))

d/dz σ(z) = d/dz [1/(1 + e^(-z))]
          = d/dz [(1 + e^(-z))^(-1)]
          = -1 × (1 + e^(-z))^(-2) × d/dz(1 + e^(-z))
          = -1 × (1 + e^(-z))^(-2) × (-e^(-z))
          = e^(-z) / (1 + e^(-z))^2
```

We can simplify this:
```
σ'(z) = e^(-z) / (1 + e^(-z))^2
      = [1/(1 + e^(-z))] × [e^(-z)/(1 + e^(-z))]
      = σ(z) × [e^(-z)/(1 + e^(-z))]
      = σ(z) × [(1 + e^(-z) - 1)/(1 + e^(-z))]
      = σ(z) × [1 - 1/(1 + e^(-z))]
      = σ(z) × [1 - σ(z)]
      = σ(z)(1 - σ(z))
```

**Key Result**: `σ'(z) = σ(z)(1 - σ(z)) = a(1 - a)`

## Step 2: Derivative of Cost Function w.r.t. Predictions (a)

The cost function for one sample is:
```
J_i = -[y_i × log(a_i) + (1 - y_i) × log(1 - a_i)]
```

Taking the derivative w.r.t. a_i:
```
∂J_i/∂a_i = -[y_i × (1/a_i) + (1 - y_i) × (1/(1 - a_i)) × (-1)]
           = -[y_i/a_i - (1 - y_i)/(1 - a_i)]
           = -[y_i(1 - a_i) - a_i(1 - y_i)] / [a_i(1 - a_i)]
           = -[y_i - y_i×a_i - a_i + a_i×y_i] / [a_i(1 - a_i)]
           = -[y_i - a_i] / [a_i(1 - a_i)]
           = (a_i - y_i) / [a_i(1 - a_i)]
```

For the full cost function (average over m samples):
```
∂J/∂a_i = (1/m) × (a_i - y_i) / [a_i(1 - a_i)]
```

## Step 3: Derivative of Cost Function w.r.t. z (Chain Rule)

Using the chain rule:
```
∂J/∂z_i = (∂J/∂a_i) × (∂a_i/∂z_i)
        = [(1/m) × (a_i - y_i) / [a_i(1 - a_i)]] × [a_i(1 - a_i)]
        = (1/m) × (a_i - y_i)
```

**Key Simplification**: The terms `a_i(1 - a_i)` cancel out!

For all samples in vector form:
```
∂J/∂z = (1/m) × (a - y)
```

## Step 4: Derivative w.r.t. Weights (Chain Rule)

Since `z = X·W + b`, we have `∂z_i/∂W_j = X_{i,j}` (the j-th feature of the i-th sample).

Using the chain rule:
```
∂J/∂W_j = Σ_i (∂J/∂z_i) × (∂z_i/∂W_j)
        = Σ_i [(1/m) × (a_i - y_i)] × X_{i,j}
        = (1/m) × Σ_i X_{i,j} × (a_i - y_i)
```

In matrix form:
```
∂J/∂W = (1/m) × X^T × (a - y)
```

Where X^T is the transpose of X, so X^T × (a - y) performs the summation over samples.

## Step 5: Derivative w.r.t. Bias

Since `z_i = Σ_j X_{i,j} × W_j + b`, we have `∂z_i/∂b = 1`.

Using the chain rule:
```
∂J/∂b = Σ_i (∂J/∂z_i) × (∂z_i/∂b)
      = Σ_i [(1/m) × (a_i - y_i)] × 1
      = (1/m) × Σ_i (a_i - y_i)
```

## Final Gradient Formulas

The gradient formulas we derived are:

1. **Weight gradients**: `dW = (1/m) × X^T × (a - y)`
2. **Bias gradient**: `db = (1/m) × Σ(a - y)`

## Why This Works So Elegantly

The beautiful simplification `∂J/∂z = (1/m) × (a - y)` occurs because:

1. The logarithmic cost function creates terms like `1/a` and `1/(1-a)`
2. The sigmoid derivative creates the term `a(1-a)`  
3. These terms cancel exactly, leaving the clean difference `(a - y)`

This is why logistic regression with cross-entropy loss and sigmoid activation is so mathematically elegant - the gradients have a very simple form that directly measures the error between predictions and true labels.

## Intuition

The gradient `(a - y)` makes intuitive sense:
- When `a > y` (overconfident positive prediction), gradient is positive → decrease weights
- When `a < y` (underconfident prediction), gradient is negative → increase weights  
- The magnitude of `|a - y|` determines how much to adjust

The factor `X^T` in the weight gradient means features that contributed more to wrong predictions get larger weight adjustments.