import os
import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image

# Suppress unnecessary TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set the model name and device
MODEL_NAME = 'bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load Model 1
model_1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model_1.load_state_dict(torch.load('DL_model_1.pth', map_location=device))
model_1.to(device)

# Load Model 3
model_3 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model_3.load_state_dict(torch.load('DL_model_3.pth', map_location=device))
model_3.to(device)


# Prediction function
def predict_review(model1, model3, tokenizer, device, review, max_len=160):
    model1.eval()
    model3.eval()

    with torch.no_grad():
        encoding_reviews = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )

        # Move tensors to the same device as the models
        inputs = {key: tensor.to(device) for key, tensor in encoding_reviews.items()}

        # Get predictions from both models
        outputs_1 = model1(**inputs)
        outputs_3 = model3(**inputs)
        pred_1 = outputs_1.logits.softmax(dim=1)
        pred_3 = outputs_3.logits.softmax(dim=1)

        # Average predictions and determine confidence
        avg_pred = (pred_1 + pred_3) / 2
        confidence, label = avg_pred.max(dim=1)
        return label.item(), confidence.item() * 100


# Streamlit App
def main():
    # Page configuration
    st.set_page_config(page_title="Clear Critique", page_icon="💬", layout="centered")

    # Title and header
    st.title("💬 Clear Critique ")
    st.markdown("""
        Welcome to the **Review Classification App**!  
        Upload your review below to determine if it's **genuine** or **generated**.
    """)

    # Add an image at the top
    try:
        image = Image.open("review.png")  # Replace with an actual image file in your directory
        st.image(image, caption="Analyze Reviews Seamlessly", use_column_width=True)
    except FileNotFoundError:
        st.warning("Image not found. Please ensure 'review_image.jpg' is in the app directory.")

    # Input section
    st.subheader("Enter Review")
    review_text = st.text_area(
        "Paste your review below and click the button to analyze:",
        placeholder="Type your review here..."
    )

    # Analyze button
    if st.button("Analyze Review"):
        if not review_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner('Analyzing...'):
                # Perform prediction
                prediction, confidence = predict_review(model_1, model_3, tokenizer, device, review_text)

                # Display results
                st.markdown("### Results")

                # Create columns for better layout
                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.success("✅ Customer Review")
                    else:
                        st.error("🤖 Computer Generated")

                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")

    # Footer
    st.markdown("---")
    st.markdown("**Built with  ❤️ using BERT .**")


if __name__ == "__main__":
    main()
