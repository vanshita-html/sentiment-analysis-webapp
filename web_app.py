import streamlit as st
import joblib

# 1. Load the model
# We use @st.cache_resource so the model loads only once, 
# making the app faster on subsequent runs.
@st.cache_resource
def load_model():
    return joblib.load('sentiment_model.pkl')

model = load_model()

# 2. Create the Title and Subtitle
st.title("🤖 AI Sentiment Analyzer")
st.write("Enter a movie review or any sentence below to detect its emotional tone.")

# 3. Create the Input Field
user_input = st.text_area("Type your text here:", height=100)

# 4. Create the Button and Logic
if st.button("Analyze Sentiment"):
    if user_input:
        # Get prediction and probability
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        confidence = max(probabilities) * 100
        
        # Display results with some styling
        st.divider()
        if prediction == 'positive':
            st.success(f"**Sentiment:** POSITIVE 😊")
        else:
            st.error(f"**Sentiment:** NEGATIVE 😠")
            
        st.info(f"**Confidence Score:** {confidence:.2f}%")
    else:
        st.warning("Please enter some text first.")