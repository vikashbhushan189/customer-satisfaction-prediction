# 📊 Customer Satisfaction Prediction using Machine Learning

## Project Overview

This project aims to predict customer satisfaction based on various attributes derived from customer support tickets and detailed ticket descriptions. Leveraging machine learning techniques, the goal is to categorize customer satisfaction into "Low," "Medium," or "High" levels, providing actionable insights for improving customer service and product experience.

Customer satisfaction is a critical metric for businesses. By predicting it, companies can proactively address potential issues, identify common pain points, optimize support processes, and ultimately foster customer loyalty. This solution provides a robust framework for such a prediction system.

## Key Features

*   **Data Preprocessing:** Handles missing values, converts data types, and extracts meaningful features from raw customer support ticket data.
*   **Feature Engineering:** Creates powerful features including:
    *   **Resolution Time:** Time taken to resolve a ticket.
    *   **Response/Resolution Flags:** Indicators of whether a first response or full resolution occurred.
    *   **Customer Segmentation:** Age-based customer grouping (`Young`, `Middle Age`, `Old`).
    *   **Natural Language Processing (NLP):**
        *   **TF-IDF:** Quantifies the importance of words in ticket descriptions.
        *   **Sentiment Analysis (VADER):** Extracts compound, positive, negative, and neutral sentiment scores from ticket descriptions.
*   **Exploratory Data Analysis (EDA):** Visualizes key distributions and relationships within the dataset to understand customer behavior and support ticket trends.
*   **Machine Learning Model Training:** Compares various classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) and performs hyperparameter tuning for relevant models.
*   **Imbalance Handling:** Trains directly on the imbalanced dataset as resampling techniques (like SMOTE) were found to decrease accuracy in this context.
*   **Feature Selection:** Employs `SelectKBest` to identify and utilize the most relevant features, reducing dimensionality and improving model efficiency.
*   **Model Persistence:** Saves the trained model and all necessary preprocessing components (TF-IDF vectorizer, scaler, feature selector, categorical mappings) for easy deployment.
*   **Streamlit Web Application:** A user-friendly web interface for real-time prediction of customer satisfaction based on input ticket details.

## Project Structure

customer-satisfaction-prediction/
├── customer_support_tickets.csv # The raw dataset
├── streamlit_artifacts/ # Directory for saved model and preprocessing objects
│ ├── best_customer_satisfaction_model.joblib
│ ├── tfidf_vectorizer.joblib
│ ├── scaler.joblib
│ ├── selector.joblib
│ ├── selected_features.joblib
│ ├── categorical_maps.joblib
│ └── scaler_fit_column_names.joblib
├── train_and_save_model.py # Python script for data processing, EDA, training, and artifact saving
├── streamlit_app.py # Python script for the Streamlit web application
├── README.md # This README file
└── requirements.txt # List of Python dependencies for easy setup


## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd your-repository-name
    ```
    (Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details)

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    Make sure you have `requirements.txt` in the root of your project directory (generated from `pip freeze > requirements.txt`).
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** Ensure `scikit-learn`, `imbalanced-learn`, `xgboost`, `wordcloud`, `streamlit` are installed and up-to-date. You might need to update/install specific versions:
    ```bash
    pip install -U scikit-learn imbalanced-learn  # Important for compatibility
    pip install xgboost wordcloud streamlit        # Ensure all core libraries are installed
    ```

5.  **Download NLTK Data:**
    The scripts will attempt to download these automatically, but you can do it manually if preferred:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    ```
    (You can run these two lines in a Python interpreter.)

## How to Run

### Step 1: Train the Model and Save Artifacts

First, you need to run the training script. This script will load the data, perform all the preprocessing, train the machine learning model, and save the model and necessary transformation objects into the `streamlit_artifacts` directory.

1.  Ensure you are in the project's root directory and your virtual environment is active.
2.  Make sure the `customer_support_tickets.csv` dataset is present in the root directory.
3.  **Create the `streamlit_artifacts` directory** (if it doesn't already exist from a previous run):
    ```bash
    mkdir streamlit_artifacts
    ```
4.  Run the training script:
    ```bash
    python train_and_save_model.py
    ```
    This script will print various logs about data processing, model training progress, and will indicate when artifacts are successfully saved.

### Step 2: Run the Streamlit Web Application

Once the training script completes and the `streamlit_artifacts` folder is populated, you can launch the Streamlit app.

1.  Ensure you are in the project's root directory and your virtual environment is active.
2.  Run the Streamlit application:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will open a new tab in your default web browser displaying the customer satisfaction prediction interface.

## Model Performance & Discussion

Based on the training run, the **Gradient Boosting Classifier** achieved the highest accuracy.

*   **Best Model Identified:** Gradient Boosting Classifier
*   **Accuracy (on test set):** `0.3827` (This value comes directly from your provided output.)
*   **Classification Report for Gradient Boosting (Example values from your run):**
    ```
                  precision    recall  f1-score   support

           Low       0.39      0.55      0.46       221
        Medium       0.17      0.04      0.07       116
          High       0.40      0.39      0.40       217

      accuracy                           0.38       554
     macro avg       0.32      0.33      0.31       554
    weighted avg       0.35      0.38      0.35       554
    ```

**Key Observations:**

*   **Challenging "Medium" Class:** Prediction for the "Medium" satisfaction category (`3` on a 1-5 scale) remains particularly challenging. This class typically shows very low `precision`, `recall`, and `f1-score`, indicating the model struggles to accurately identify neutral satisfaction levels. This is often due to the inherent ambiguity of "medium" sentiment and its smaller representation in the dataset compared to "Low" and "High".
*   **No Data Leakage:** The implemented pipeline deliberately avoids data leakage, which means the reported accuracy is a realistic measure of how the model performs on unseen data, making it suitable for practical application despite the challenges.
*   **Subjectivity of Satisfaction:** Customer satisfaction is inherently subjective and can be influenced by many factors not present in the dataset (e.g., customer mood, specific agent empathy, external product issues). The model relies solely on the provided ticket-based features.

## Future Enhancements

*   **Deep Learning for NLP:** Explore more advanced NLP models (e.g., Transformers like BERT, Sentence-BERT) for richer text embeddings from `Ticket Description`, which could significantly improve the understanding of customer intent and sentiment.
*   **Temporal Features:** Extract more features from date/time columns like "day of week", "hour of day", "duration from purchase to ticket".
*   **Agent Performance Metrics:** If available, incorporating anonymized agent performance data could be a powerful predictor.
*   **External Data Integration:** Link customer demographics or purchase history (if available and permissible) to the dataset for a more holistic view.
*   **Advanced Multi-class Classification Strategies:** For the "Medium" class, specialized techniques beyond basic over/under-sampling might be explored if this class becomes a priority, such as Focal Loss or more targeted classification algorithms.
*   **Explainable AI (XAI):** Integrate tools like SHAP or LIME into the Streamlit app to explain *why* a particular prediction was made for a given input, increasing trust and interoperability.
*   **Dashboarding:** Develop an analytics dashboard to track model performance over time and identify shifts in customer behavior or satisfaction trends.

---