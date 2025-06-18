import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("🔍 Spam Detector")
print("Type a message to check if it's spam (type 'exit' to quit)\n")

while True:
    message = input("Your message: ")

    if message.lower() == 'exit':
        print("Goodbye!")
        break

    # Predict
    prediction = model.predict([message])[0]

    label = "🚨 Spam" if prediction == 1 else "✅ Not Spam"
    print(f"Result: {label}\n")
