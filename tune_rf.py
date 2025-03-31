# To install required packages, run:
# pip install numpy scikit-learn matplotlib

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def collatz_stopping_time(n):
    """Calculate the stopping time for a given number in the Collatz sequence."""
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps

def create_features(n):
    """Create features for the number n that might be relevant for predicting stopping time."""
    return np.array([
        n,                   # The number itself
        np.log2(n),          # Logarithm base 2
        np.log(n),           # Natural logarithm
        np.log10(n),         # Logarithm base 10
        np.log2(n)/4,        # Approximation of log base 16
        np.log2(n)/5,        # Approximation of log base 32
        np.log2(n)/6,        # Approximation of log base 64
        np.log2(n)/7,        # Approximation of log base 128
        np.log2(n)/8,        # Approximation of log base 256
        np.log2(n)/9,        # Approximation of log base 512
        np.log2(n)/10,       # Approximation of log base 1024
        n % 2,               # Parity
        n % 4,               # Modulo 4
        n % 8,               # Modulo 8
        n % 16,              # Modulo 16
        bin(n).count('1'),   # Number of 1s in binary representation
        len(bin(n)) - 2,     # Length of binary representation
        len(str(n))          # Length of decimal representation
    ])

def generate_training_data(max_n=1000000):
    """Generate training data for numbers up to max_n."""
    X = []
    y = []
    
    for n in range(1, max_n + 1):
        X.append(create_features(n))
        y.append(collatz_stopping_time(n))
    
    return np.array(X), np.array(y)

def main():
    # Generate training data
    print("Generating training data...")
    X, y = generate_training_data(max_n=1000000)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define a more focused parameter grid for GridSearchCV
    # Reduced number of combinations to speed up search
    param_grid = {
        'n_estimators': [100, 200],       # Reduced options
        'max_depth': [20, None],          # Reduced options
        'min_samples_split': [2, 5],      # Reduced options
        'min_samples_leaf': [1, 2],       # Reduced options
        'max_features': ['sqrt', None]    # Reduced options
    }
    
    # Create base Random Forest model
    base_rf = RandomForestRegressor(random_state=42)
    
    # Create GridSearchCV object with reduced CV folds
    print("Performing hyperparameter optimization...")
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=2,                # Reduced from 3 to 2 folds
        n_jobs=-1,           # Use all available cores
        scoring='r2',
        verbose=1
    )
    
    # Use a subset of training data for faster hyperparameter tuning
    # This significantly reduces search time while still finding good parameters
    subset_size = min(100000, len(X_train))  # Use at most 100k samples
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    
    # Fit GridSearchCV on the subset
    grid_search.fit(X_train_subset, y_train_subset)
    
    # Print best parameters
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nBest cross-validation R² score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    rf = grid_search.best_estimator_
    
    # Make predictions with the best model
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # Calculate comprehensive metrics
    print("\nModel Performance Metrics:")
    print("Training Set:")
    print(f"R² Score: {r2_score(y_train, y_pred_train):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    
    print("\nTest Set:")
    print(f"R² Score: {r2_score(y_test, y_pred_test):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    # Feature importance
    feature_names = [
        'Number', 'Log2', 'LogN', 'Log10', 'Log16', 'Log32', 
        'Log64', 'Log128', 'Log256', 'Log512', 'Log1024',
        'Parity', 'Mod4', 'Mod8', 'Mod16', 'Binary1s', 'BinaryLength',
        'DecimalLength'
    ]
    importances = rf.feature_importances_
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), importances)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.title('Feature Importance in Predicting Collatz Stopping Times')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Make some predictions
    test_numbers = [100, 1000, 10000, 100000]
    print("\nPredictions for test numbers:")
    for n in test_numbers:
        features = create_features(n)
        actual = collatz_stopping_time(n)
        predicted = rf.predict([features])[0]
        print(f"Number: {n}")
        print(f"Actual stopping time: {actual}")
        print(f"Predicted stopping time: {predicted:.2f}")
        print(f"Error: {abs(actual - predicted):.2f}\n")
    
    # Plot the loss function (prediction errors)
    plt.figure(figsize=(10, 6))
    train_errors = np.abs(y_train - y_pred_train)
    test_errors = np.abs(y_test - y_pred_test)
    
    plt.hist(train_errors, alpha=0.5, bins=50, label='Training Errors')
    plt.hist(test_errors, alpha=0.5, bins=50, label='Testing Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_distribution.png')
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Stopping Time')
    plt.ylabel('Predicted Stopping Time')
    plt.title('Actual vs Predicted Stopping Times')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    
    # Plot predictions vs real stopping times with x as input
    plt.figure(figsize=(12, 8))
    
    # Generate a range of numbers to plot (using logarithmic spacing for 1e6 range)
    plot_range = np.unique(np.logspace(0, 6, num=1000, dtype=int))
    
    # Calculate actual stopping times
    print("Calculating actual stopping times for large range...")
    actual_stopping_times = [collatz_stopping_time(n) for n in plot_range]
    
    # Predict stopping times
    print("Predicting stopping times for large range...")
    predicted_stopping_times = []
    for n in plot_range:
        features = create_features(n)
        predicted = rf.predict([features])[0]
        predicted_stopping_times.append(predicted)
    
    # Plot both curves
    plt.plot(plot_range, actual_stopping_times, 'b-', alpha=0.7, label='Actual Stopping Times')
    plt.plot(plot_range, predicted_stopping_times, 'r-', alpha=0.7, label='Predicted Stopping Times')
    
    plt.xlabel('Number (log scale)')
    plt.ylabel('Stopping Time')
    plt.title('Collatz Stopping Times: Actual vs Predicted (up to 1e6)')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stopping_times_comparison_large.png')

if __name__ == "__main__":
    main()