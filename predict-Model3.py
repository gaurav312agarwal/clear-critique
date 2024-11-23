import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def load_model():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set the model name
    MODEL_NAME = 'bert-base-uncased'
    
    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)
    
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load('DL_model_3.pth', map_location=torch.device('cpu'), weights_only=True))
    except FileNotFoundError:
        st.error("Model file 'DL_model_3.pth' not found. Please ensure it's in the same directory as this script.")
        st.stop()
    
    return model, tokenizer, device

def predict_review(model, tokenizer, device, review_text, max_len=160):
    model.eval()
    
    with torch.no_grad():
        encoding_reviews = tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        outputs = model(**encoding_reviews.to(device))
        predictions = outputs.logits.argmax(dim=1)
        
        # Get prediction probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = probs.max().item() * 100
        
    return predictions.item(), confidence

def main():
    st.set_page_config(
        page_title="Review Authenticity Checker",
        page_icon="🔍",
        layout="centered"
    )
    
    st.title("Review Authenticity Checker 🤖")
    st.write("""
    This application helps detect whether a review was written by a real customer
    or generated by a computer. Simply enter the review text below and click 'Analyze'.
    """)
    
    # Load model on first run
    if 'model' not in st.session_state:
        with st.spinner('Loading model... Please wait...'):
            model, tokenizer, device = load_model()
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['device'] = device
    
    # Text input area
    review_text = st.text_area(
        "Enter the review text:",
        height=150,
        placeholder="Enter the review text here..."
    )
    
    # Analysis button
    if st.button("Analyze Review"):
        if not review_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner('Analyzing...'):
                prediction, confidence = predict_review(
                    st.session_state['model'],
                    st.session_state['tokenizer'],
                    st.session_state['device'],
                    review_text
                )
                
                # Display results with appropriate styling
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
                
             

if __name__ == "__main__":
    main()