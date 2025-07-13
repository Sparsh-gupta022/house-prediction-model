import numpy as np

# Random data generation
np.random.seed(1)

# 100 houses ke liye generate kr rhe h (1000–3500 sq ft)
area = np.random.randint(1000, 3500, 100)

# 200rs/sq.ft ke liye prices generate
price = 200 * area + np.random.randint(-60000, 60000, 100)

# Initialize model parameters
w = 0  # weight
b = 0  # bias

# Training settings
lr = 0.00000015  
epochs = 400     

#  Gradient Descent ki training
for i in range(epochs):
    y_pred = w * area + b
    error = y_pred - price
    loss = np.mean(error ** 2)

    dw = np.mean(2 * error * area)
    db = np.mean(2 * error)

    w -= lr * dw
    b -= lr * db

    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss:.2f}")

# Print learned parameters
print(f"\nTrained weight (w): {w:.2f}")
print(f"Trained bias (b): {b:.2f}")

# Predicting rate of a 2000 sq ft house
area_input = 2000
predicted_price = w * area_input + b
print(f"\nPredicted price for {area_input} sq ft house: ₹{predicted_price:.0f}")

# Calculating Mean Absolute Percentage Error
y_pred_all = w * area + b
mape = np.mean(np.abs((price - y_pred_all) / price)) * 100
print(f"\nModel error (MAPE): {mape:.2f}%")  # it should show almost 18% error
