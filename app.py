import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model("spam_model.h5")
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100

st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Spam Detection")
st.markdown("Detect whether a message is **spam** or **not spam** using an AI model trained on SMS messages.")

st.markdown("---")

user_input = st.text_area("Enter your message:", height=150)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to detect.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.error("🚫 This is likely **SPAM**.")
        else:
            st.success("✅ This is likely **NOT SPAM**.")

with st.sidebar:
    st.header("🧠 Model Info")
    st.write("• Trained on SMS Spam Collection dataset")
    st.write("• Deep Learning model using TensorFlow/Keras")
    st.markdown("---")
    st.header("📊 Accuracy")
    st.write("Around 95% on validation data")
    st.markdown("---")
    st.write("Made with ❤️ by [@theidealcoder](https://www.instagram.com/theidealcoder)")
