import joblib

# 1. Load the saved model
# We don't need to retrain it! We just load the "brain" we saved earlier.
model = joblib.load('sentiment_model.pkl')

def predict_sentiment(text):
    # The model expects a list of texts, so we wrap the input in a list
    prediction = model.predict([text])[0]
    
    # Get the probability (confidence) of the prediction
    # This shows how sure the AI is about its answer
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities) * 100
    
    return prediction, confidence

if __name__ == "__main__":
    print("--- AI Sentiment Analysis Tool ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Enter a sentence: ")
        
        if user_input.lower() == 'exit':
            break
            
        sentiment, confidence = predict_sentiment(user_input)
        
        print(f"Prediction: {sentiment.upper()}")
        print(f"Confidence: {confidence:.2f}%\n")