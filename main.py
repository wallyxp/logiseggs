from logisticregressionnn import LogisticRegressionNN
from create_dataset import generate_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Generate data
    X_train, X_test, y_train, y_test = generate_data()
    
    # Standardize features (important for gradient descent)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # Create and train model
    model = LogisticRegressionNN(learning_rate=0.1, max_iterations=2000)
    
    print("Training Logistic Regression Neural Network...")
    model.fit(X_train_scaled, y_train, verbose=True)
    
    # Make predictions
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Print learned parameters
    print(f"\nLearned Parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    
    # Plot cost function
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.cost_history)
    plt.title('Cost Function During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    h = 0.02  # step size in mesh
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.colorbar(label='Probability')
    
    # Plot training points
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                         c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('Decision Boundary and Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    # Example predictions on new data
    print(f"\nExample Predictions:")
    sample_indices = np.random.choice(len(X_test_scaled), 5)
    for i in sample_indices:
        prob = model.predict_proba(X_test_scaled[i:i+1])[0]
        pred = model.predict(X_test_scaled[i:i+1])[0]
        actual = y_test[i]
        print(f"Sample {i}: Probability={prob:.4f}, Predicted={pred}, Actual={actual}")