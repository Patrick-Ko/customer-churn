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
st.write("This app provides two prediction options:")
st.write("- **Bulk Prediction**: Upload a CSV file containing customer data for batch predictions.")
st.write("- **Single Prediction**: Manually enter customer details on the site for individual predictions.")
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

        st.markdown("---")
        st.header("Manual Input for Single Prediction")

        # Create input fields for each feature with updated hints and user-friendly options
        account_length = st.number_input("Account Length (in months)", min_value=0, help="Enter the duration of the account in months.")
        local_calls = st.number_input("Local Calls", min_value=0, help="Enter the number of local calls made by the customer.")
        local_mins = st.number_input("Local Mins", min_value=0.0, help="Enter the total minutes of local calls.")
        intl_calls = st.number_input("Intl Calls", min_value=0, help="Enter the number of international calls made by the customer.")
        intl_mins = st.number_input("Intl Mins", min_value=0.0, help="Enter the total minutes of international calls.")
        extra_intl_charges = st.number_input("Extra International Charges", min_value=0.0, help="Enter the additional charges for international calls.")
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, help="Enter the number of calls made to customer service.")
        avg_monthly_gb_download = st.number_input("Avg Monthly GB Download", min_value=0.0, help="Enter the average monthly data usage in GB.")
        extra_data_charges = st.number_input("Extra Data Charges", min_value=0.0, help="Enter the additional charges for extra data usage.")
        age = st.number_input("Age", min_value=0, help="Enter the age of the customer.")
        num_customers_in_group = st.number_input("Number of Customers in Group", min_value=0, help="Enter the number of customers in the group.")
        monthly_charge = st.number_input("Monthly Charge", min_value=0.0, help="Enter the monthly charge for the customer.")
        total_charges = st.number_input("Total Charges", min_value=0.0, help="Enter the total charges incurred by the customer.")
        intl_active_yes = st.selectbox("Active International Plan", ["No", "Yes"], help="Select whether the customer has an active international plan.")
        intl_plan_yes = st.selectbox("International Plan", ["No", "Yes"], help="Select whether the customer has an international plan.")
        unlimited_data_plan_yes = st.selectbox("Unlimited Data Plan", ["No", "Yes"], help="Select whether the customer has an unlimited data plan.")
        gender_male = st.selectbox("Gender", ["Female", "Male"], help="Select the gender of the customer.")
        gender_prefer_not_to_say = st.selectbox("Prefer Not to Say Gender", ["No", "Yes"], help="Select whether the customer prefers not to disclose gender.")
        under_30_yes = st.selectbox("Under 30", ["No", "Yes"], help="Select whether the customer is under 30 years old.")
        senior_yes = st.selectbox("Senior Citizen", ["No", "Yes"], help="Select whether the customer is a senior citizen.")
        group_yes = st.selectbox("Belongs to Group", ["No", "Yes"], help="Select whether the customer belongs to a group.")
        device_protection_online_backup_yes = st.selectbox("Device Protection & Online Backup", ["No", "Yes"], help="Select whether the customer has device protection and online backup.")
        contract_type_one_year = st.selectbox("One-Year Contract", ["No", "Yes"], help="Select whether the customer has a one-year contract.")
        contract_type_two_year = st.selectbox("Two-Year Contract", ["No", "Yes"], help="Select whether the customer has a two-year contract.")
        payment_method_direct_debit = st.selectbox("Payment Method: Direct Debit", ["No", "Yes"], help="Select whether the payment method is direct debit.")
        payment_method_paper_check = st.selectbox("Payment Method: Paper Check", ["No", "Yes"], help="Select whether the payment method is paper check.")

        # Map user-friendly inputs to model-compatible values
        input_data = pd.DataFrame({
            'Account Length (in months)': [account_length],
            'Local Calls': [local_calls],
            'Local Mins': [local_mins],
            'Intl Calls': [intl_calls],
            'Intl Mins': [intl_mins],
            'Extra International Charges': [extra_intl_charges],
            'Customer Service Calls': [customer_service_calls],
            'Avg Monthly GB Download': [avg_monthly_gb_download],
            'Extra Data Charges': [extra_data_charges],
            'Age': [age],
            'Number of Customers in Group': [num_customers_in_group],
            'Monthly Charge': [monthly_charge],
            'Total Charges': [total_charges],
            'Intl Active_Yes': [1 if intl_active_yes == "Yes" else 0],
            'Intl Plan_yes': [1 if intl_plan_yes == "Yes" else 0],
            'Unlimited Data Plan_Yes': [1 if unlimited_data_plan_yes == "Yes" else 0],
            'Gender_Male': [1 if gender_male == "Male" else 0],
            'Gender_Prefer not to say': [1 if gender_prefer_not_to_say == "Yes" else 0],
            'Under 30_Yes': [1 if under_30_yes == "Yes" else 0],
            'Senior_Yes': [1 if senior_yes == "Yes" else 0],
            'Group_Yes': [1 if group_yes == "Yes" else 0],
            'Device Protection & Online Backup_Yes': [1 if device_protection_online_backup_yes == "Yes" else 0],
            'Contract Type_One Year': [1 if contract_type_one_year == "Yes" else 0],
            'Contract Type_Two Year': [1 if contract_type_two_year == "Yes" else 0],
            'Payment Method_Direct Debit': [1 if payment_method_direct_debit == "Yes" else 0],
            'Payment Method_Paper Check': [1 if payment_method_paper_check == "Yes" else 0],
        })

        # Predict churn when the user clicks the button
        if st.button("Predict Churn"):
            try:
                # Load the model
                model = joblib.load("model.joblib")

                # Make prediction
                prediction = model.predict(input_data)[0]

                # Display result
                result = "Churn" if prediction == 1 else "Not Churn"
                st.success(f"The customer is predicted to: {result}")
            except Exception as e:
                st.error(f"An error occurred: {e}")