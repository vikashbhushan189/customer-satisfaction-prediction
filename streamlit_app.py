import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer # For VADER


# --- Helper Functions (Copied from training script) ---
# Clean Text Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Ensure NLTK resources are downloaded for sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    st.info("Downloading NLTK 'vader_lexicon' for sentiment analysis...")
    nltk.download('vader_lexicon', quiet=True)

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()


# --- Load Preprocessing Artifacts and Model ---
@st.cache_resource # Use Streamlit's caching to load these heavy objects only once
def load_artifacts():
    try:
        model = joblib.load('streamlit_artifacts/best_customer_satisfaction_model.joblib')
        tfidf = joblib.load('streamlit_artifacts/tfidf_vectorizer.joblib')
        scaler = joblib.load('streamlit_artifacts/scaler.joblib')
        selector = joblib.load('streamlit_artifacts/selector.joblib')
        selected_features_names = joblib.load('streamlit_artifacts/selected_features.joblib')
        categorical_maps = joblib.load('streamlit_artifacts/categorical_maps.joblib')
        scaler_fit_column_names = joblib.load('streamlit_artifacts/scaler_fit_column_names.joblib') # Load names scaler was fitted on

        # Define reverse label mapping for output display directly in the app
        reverse_label_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

        return model, tfidf, scaler, selector, selected_features_names, categorical_maps, scaler_fit_column_names, reverse_label_mapping
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}. Make sure you have run 'train_and_save_model.py' first and the 'streamlit_artifacts' folder exists with all files.")
        st.stop() # Stop the app if artifacts are missing
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}. Please check your saved files.")
        st.stop()


model, tfidf_vectorizer, scaler, selector, selected_features_names, categorical_maps, scaler_fit_column_names, reverse_label_mapping = load_artifacts()

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Satisfaction Predictor", layout="wide", initial_sidebar_state="auto")

st.title("💡 Customer Satisfaction Prediction")
st.markdown("Enter the customer support ticket details to predict the likely satisfaction level.")
st.markdown("""
    This application uses a trained Machine Learning model to predict customer satisfaction based on various ticket attributes and description.
    Please fill in the details below:
""")

# Input Sections
st.header("Ticket Details")

# Layout with columns for better organization
col1, col2, col3 = st.columns(3)

with col1:
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30, help="Age of the customer.")
    # Assuming the available genders from your categorical_maps
    customer_gender = st.selectbox("Customer Gender", list(categorical_maps.get('Customer Gender', ['Male', 'Female', 'Other']).keys()), help="Gender of the customer.")
    product_purchased = st.selectbox("Product Purchased", list(categorical_maps.get('Product Purchased', ['Laptop', 'Mobile', 'Other']).keys()), help="Product the customer purchased.")

with col2:
    ticket_type = st.selectbox("Ticket Type", list(categorical_maps.get('Ticket Type', ['Technical issue', 'Billing inquiry', 'Product inquiry']).keys()), help="Category of the support ticket.")
    ticket_channel = st.selectbox("Ticket Channel", list(categorical_maps.get('Ticket Channel', ['Email', 'Phone', 'Chat', 'Social media']).keys()), help="Channel through which the ticket was raised.")
    ticket_priority = st.selectbox("Ticket Priority", list(categorical_maps.get('Ticket Priority', ['Low', 'Medium', 'High', 'Critical']).keys()), help="Priority level assigned to the ticket.")

with col3:
    ticket_subject = st.selectbox("Ticket Subject", list(categorical_maps.get('Ticket Subject', ['Product setup', 'Software bug', 'Account access']).keys()), help="Topic/subject of the ticket.")
    # Placeholders for future calculated/estimated values if not directly input
    st.markdown("##") # Space out inputs visually
    interval_hours = st.number_input("Resolution Time (Hours)", min_value=0.0, max_value=720.0, value=24.0, format="%.1f", help="Estimated or actual time taken to resolve the ticket in hours.")
    has_response = st.selectbox("Was there a First Response?", ['Yes', 'No'], index=0, help="Indicates if a first response was provided. For prediction, assumes 'Yes'.")
    # 'Has_Resolution' is always 'Yes' for prediction context as we are predicting satisfaction *for resolved tickets*.


st.header("Ticket Description")
ticket_description = st.text_area("Detailed Ticket Description",
                                  "My device is not turning on after the latest software update and it keeps crashing. I've tried restarting multiple times.",
                                  height=150, help="Provide a detailed description of the customer's issue or inquiry.")

st.markdown("---")

if st.button("Predict Satisfaction"):
    st.info("Predicting customer satisfaction... Please wait.")
    try:
        # --- 1. Prepare initial input into a DataFrame with core features ---
        input_df = pd.DataFrame({
            'Customer Age': [customer_age],
            'Customer Gender': [customer_gender],
            'Product Purchased': [product_purchased],
            'Ticket Type': [ticket_type],
            'Ticket Channel': [ticket_channel],
            'Ticket Priority': [ticket_priority],
            'Ticket Subject': [ticket_subject],
            'Has_Response': [1 if has_response == 'Yes' else 0],
            'Has_Resolution': [1], # Always 1 as we are predicting for resolved tickets
            'Interval_Hours': [interval_hours]
        })

        # --- 2. Apply preprocessing steps in the correct order ---

        # 2a. Derived feature: 'Type of Customer'
        age = input_df['Customer Age'].iloc[0]
        if age <= 30:
            input_df.loc[0, 'Type of Customer'] = 'Young Customer'
        elif 30 < age < 55:
            input_df.loc[0, 'Type of Customer'] = 'Middle Age Customer'
        else:
            input_df.loc[0, 'Type of Customer'] = 'Old Customer'


        # 2b. Categorical Encoding (manual mapping from loaded maps)
        # Apply encoding to columns in input_df that have a mapping
        for col_name, mapping_dict in categorical_maps.items():
            if col_name in input_df.columns:
                # Handle cases where user selects a category that somehow isn't in mapping (though selectbox prevents this)
                # Or, if mapping values need special type handling before .get()
                current_value = input_df.loc[0, col_name]
                encoded_value = mapping_dict.get(str(current_value), -1) # Use str() for robustness
                if encoded_value == -1: # Log/handle unseen categories
                    st.warning(f"Warning: Category '{current_value}' for column '{col_name}' was not seen during training. Assigned default -1.")
                input_df.loc[0, col_name] = encoded_value
            # Note: columns not in `categorical_maps` will remain untouched if they are already numerical (like age, interval)


        # 2c. Sentiment features from Ticket Description
        cleaned_description = clean_text(ticket_description)
        sentiment_scores = sid.polarity_scores(cleaned_description)
        input_df['Sentiment_Description_Compound'] = sentiment_scores['compound']
        input_df['Sentiment_Description_Positive'] = sentiment_scores['pos']
        input_df['Sentiment_Description_Negative'] = sentiment_scores['neg']
        input_df['Sentiment_Description_Neutral'] = sentiment_scores['neu']


        # 2d. TF-IDF features from Ticket Description
        # tfidf_vectorizer was fitted on ALL descriptions during training
        tfidf_features_sparse_resolved = tfidf_vectorizer.transform([cleaned_description])
        tfidf_df_single = pd.DataFrame(tfidf_features_sparse_resolved.toarray(),
                                       columns=['tfidf_' + col for col in tfidf_vectorizer.get_feature_names_out()],
                                       index=input_df.index) # Use input_df index to match for concat


        # Construct the full DataFrame that mirrors `X` from training
        # This uses `scaler_fit_column_names` to create an empty DataFrame with ALL expected columns,
        # then populates it with our single input sample. This ensures column presence and order.
        full_feature_df = pd.DataFrame(0.0, index=[0], columns=scaler_fit_column_names)

        # Populate `full_feature_df` from `input_df` and `tfidf_df_single`
        # Using `loc` for precise updates
        for col in input_df.columns:
            if col in full_feature_df.columns:
                full_feature_df.loc[0, col] = input_df.loc[0, col]
        for col in tfidf_df_single.columns:
            if col in full_feature_df.columns:
                full_feature_df.loc[0, col] = tfidf_df_single.loc[0, col]

        # Ensure no remaining NaN values before scaling/selection, fill with 0
        full_feature_df.fillna(0, inplace=True)


        # 2e. Apply StandardScaler to the entire `full_feature_df`
        # `scaler` was fitted on ALL features in X. Now we transform all features of the input sample.
        scaled_full_feature_array = scaler.transform(full_feature_df)
        scaled_full_feature_df = pd.DataFrame(scaled_full_feature_array, columns=scaler_fit_column_names, index=[0])


        # 2f. Feature Selection: apply the SelectKBest filter
        # `selector.transform` will reduce the scaled features to only the `selected_features_names`.
        selected_features_data_array = selector.transform(scaled_full_feature_df)
        # Create final DataFrame with only the selected feature names in correct order
        final_input_for_prediction = pd.DataFrame(selected_features_data_array, columns=selected_features_names, index=[0])

        # --- 3. Make Prediction ---
        prediction_encoded = model.predict(final_input_for_prediction)
        predicted_satisfaction = reverse_label_mapping[prediction_encoded[0]]

        st.success(f"**Predicted Customer Satisfaction Level: {predicted_satisfaction}**")

        st.subheader("What do the prediction labels mean?")
        st.info("""
        - **Low (Rating 1-2):** Customer is likely dissatisfied.
        - **Medium (Rating 3):** Customer is neutral or moderately satisfied/dissatisfied.
        - **High (Rating 4-5):** Customer is highly satisfied.
        """)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and the integrity of the saved model artifacts.")

st.markdown("---")
st.subheader("Model and Preprocessing Information:")
st.markdown("""
- **Model Used:** The best performing model selected during training (typically Tuned Random Forest).
- **Feature Engineering:** Incorporates TF-IDF on ticket descriptions, VADER sentiment analysis, customer demographics (age group), ticket handling attributes (resolution time, response flags).
- **Data Preprocessing:** Includes Label Encoding for categorical variables, StandardScaler for numerical feature scaling, and SelectKBest for essential feature selection.
- **Note on Accuracy:** Predicting subjective customer satisfaction is inherently challenging. The model aims to provide the most probable outcome given the data. If the model had shown 100% accuracy in initial runs, it likely indicated **data leakage**, which has been corrected for a more realistic performance assessment.
- **Accuracy on test set (example):** ~35-40% (Actual accuracy will vary based on specific training run)
- **Specific Challenge:** Prediction for 'Medium' satisfaction is often the hardest due to its nature as a neutral point and potential class imbalance in original data.
""")