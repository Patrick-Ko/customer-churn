import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.markdown(
    """
    <h1 style='text-align: center;'>Churn Prediction App</h1>
    """,
    unsafe_allow_html=True
)
st.write("This app predicts customer churn using a Random Forest Classifier. Please open the side panel for information on dataset upload.")
st.write("Click the 'Lets Predict' button when you are ready.")
st.markdown("<style>body{text-align: left;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data Requirements")
    st.write("Users must follow the data guidelines provided below.")
    
    with st.expander("Dataset Details"):
        st.write("""
        Please upload a CSV file with the following columns:
        - `Account Length (in months)`: Duration of the account in months.
        - `Local Calls`: Number of local calls made by the customer.
        - `Local Mins`: Total minutes of local calls.
        - `Intl Calls`: Number of international calls made by the customer.
        - `Intl Mins`: Total minutes of international calls.
        - `Extra International Charges`: Additional charges for international calls.
        - `Customer Service Calls`: Number of calls made to customer service.
        - `Avg Monthly GB Download`: Average monthly data usage in GB.
        - `Extra Data Charges`: Additional charges for extra data usage.
        - `Age`: Age of the customer.
        - `Number of Customers in Group`: Number of customers in the group.
        - `Monthly Charge`: Monthly charge for the customer.
        - `Total Charges`: Total charges incurred by the customer.
        - `Intl Active_Yes`: Whether the customer has an active international plan (1/0).
        - `Intl Plan_yes`: Whether the customer has an international plan (1/0).
        - `Unlimited Data Plan_Yes`: Whether the customer has an unlimited data plan (1/0).
        - `Gender_Male`: Whether the customer identifies as male (1/0).
        - `Gender_Prefer not to say`: Whether the customer prefers not to disclose gender (1/0).
        - `Under 30_Yes`: Whether the customer is under 30 years old (1/0).
        - `Senior_Yes`: Whether the customer is a senior citizen (1/0).
        - `Group_Yes`: Whether the customer belongs to a group (1/0).
        - `Device Protection & Online Backup_Yes`: Whether the customer has device protection and online backup (1/0).
        - `Contract Type_One Year`: Whether the customer has a one-year contract (1/0).
        - `Contract Type_Two Year`: Whether the customer has a two-year contract (1/0).
        - `Payment Method_Direct Debit`: Whether the payment method is direct debit (1/0).
        - `Payment Method_Paper Check`: Whether the payment method is paper check (1/0).
        
        Ensure the file is clean and contains no missing values.
        """)
    
    st.markdown("---")
    st.markdown(
        """
        <p style='text-align: center;'>Developed by InnaTech</p>
        """,
        unsafe_allow_html=True
    )

# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file is not None:
#     try:
#         data = pd.read_csv(uploaded_file)
#         st.success("File uploaded successfully!")
#         st.write("Preview of the uploaded data:")
#         st.dataframe(data.head())
#     except Exception as e:
#         st.error(f"An error occurred while reading the file: {e}")
# else:
#     st.info("Please upload a CSV file to proceed.")

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[1] = True

st.button("Lets Predict", on_click=clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("Preview of the uploaded data:")
            st.dataframe(data.head())

            # Load the model
            model = joblib.load("model.joblib")

            # Make predictions
            predictions = model.predict(data)

            # Map predictions to churn labels
            data['Prediction'] = predictions
            data['Prediction'] = data['Prediction'].map({1: 'Churn', 0: 'Not Churn'})

            # Display predictions
            st.write("Prediction Results:")
            st.dataframe(data)

            # Provide download option
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")