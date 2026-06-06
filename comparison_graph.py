import matplotlib.pyplot as plt

# Accuracy values (you can adjust if needed)
models = ["ANN", "KNN"]
accuracy = [99.0, 97.0]   # use your real values if different

plt.figure(figsize=(6,4))

plt.bar(models, accuracy, color=['blue', 'green'])

plt.title("Model Accuracy Comparison (ANN vs KNN)")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

for i, v in enumerate(accuracy):
    plt.text(i, v + 0.5, str(v) + "%", ha='center')

plt.ylim(0, 100)

plt.show()