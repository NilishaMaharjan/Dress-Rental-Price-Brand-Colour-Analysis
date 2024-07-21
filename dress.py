import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the saved model
model_path = r'C:\Users\DELL\Desktop\Dress_Rental_Analysis\ML_MODEL\knn_model.pkl'

# Load joblib model
model = joblib.load(model_path)    

def load_data():
    # Load the dataset
    file_path  = r"C:\Users\DELL\Desktop\Dress_Rental_Analysis\dataset\dress_rental_prices.csv"
    return pd.read_csv(file_path, encoding= 'latin1')

def main():
    # Set the title of the web app
    st.title('Dress Rental Price, Brand/Color Prediction')

    # Add a description
    st.write('Predict the rental price of a dress based on various features like brand, color, and dress type.')

    # Load data
    input_data = load_data()

    
    # Define the expected features for the model
    # Adjust these features based on your model's requirements
    feature_names = ['Brand', 'Catagories', 'Colour', 'ID', 'Name']


    # Create columns for layout
    col1, col2  = st.columns(2)
    
    with col1:
        st.subheader('Dress Information')

       
        # Use selectbox for selecting brand
        Brand = st.selectbox("Brand", options=input_data['Brand'].unique())
        Catagories = st.selectbox("Catagories", ['dresses','midi','Wool-Cashmere', 'Loose','Winter', '3/4 Sleeves', 'Scoop Neck', 'Monochrome','Casual','mini','party','cotton','Short Sleeve','Maxi','Metallic','Floral','Others'])
        Colour = st.selectbox("Colour of the dress", ['beige', 'black', 'blue'])

         # Ensure the 'ID' column is not empty before using it for options
        id_options = input_data['ID'].unique() if not input_data['ID'].empty else [0]
        ID_Number = st.select_slider("ID Number", options=id_options, key="id_slider")

         # Add input fields for features
        Dress_name  = st.text_input('Dress Name')
        Price = st.number_input('Price', min_value=0, step=10)
        
        untrained_column = st.text_input('Additional Information (not used in prediction)')

        # Convert categorical inputs to numerical
        input_data['Brand_num'] = np.where(input_data['Brand'] == 'RIXO', 1, 0)
        input_data['Name_num'] = np.where(input_data['Name'] == 'BURBERRY SILK DRESS', 1, 0)
        input_data['Catagories_num'] = np.where(input_data['Catagories'] == 'dresses', 1, 0)
        input_data['Colour_num'] = np.where(input_data['Colour'] == 'black', 1, 0)
    

        
   
     # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Brand': [Brand],
        'Catagories': [Catagories],
        'Colour': [Colour],
        'ID': [ID_Number],
        'Name': [Dress_name],
        'Price': [Price] 
    })
        

    # Ensure columns are in the same order as during model training
    expected_columns = ['ID','Name', 'Brand', 'Colour','Catagories', 'Price']
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            try:
                 # Convert categorical variables to numerical variables
                input_data['Brand'] = input_data['Brand'].map(input_data['Brand'].value_counts().index)
                input_data['Catagories'] = input_data['Catagories'].map(input_data['Catagories'].value_counts().index)
                input_data['Colour'] = input_data['Colour'].map(input_data['Colour'].value_counts().index)

                # Perform prediction
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][prediction[0]]

                # Display prediction results
                st.write(f'Prediction for {Dress_name}: {"Available" if prediction[0] == 1 else "Unavailable"}')
                st.write(f'Probability of Availability: {probability:.2f}')

                price_info = (
                    input_data[input_data["Name"] == Dress_name]["Price"].values[0]
                    if Dress_name in input_data["Name"].values else "Unknown"
                )
                st.write(f'Price: {price_info}')

                # Plotting
                fig, axes = plt.subplots(3, 1, figsize=(8, 16))

                # Plot Availability/Unavailability probability
                sns.barplot(
                    x=['Unavailable', 'Available'],
                    y=[1 - probability, probability],
                    ax=axes[0],
                    palette=['red', 'green']
                )
                axes[0].set_title('Availability Probability')
                axes[0].set_ylabel('Probability')

                # Plot Price distribution
                sns.histplot(input_data['Price'], kde=True, ax=axes[1])
                axes[1].set_title('Price Distribution')

                # Plot Dress available/unavailable pie chart
                axes[2].pie(
                    [1 - probability, probability],
                    labels=['Unavailable', 'Available'],
                    autopct='%1.1f%%',
                    colors=['red', 'green']
                )
                axes[2].set_title('Dress Availability Pie Chart')

                # Display the plots
                st.pyplot(fig)

                # Provide recommendations
                if prediction[0] == 1:
                    st.success(f"{Dress_name} is available for renting. Happy shopping!!")
                else:
                    st.error(f"{Dress_name} is not available for renting. Contact us for seeking additional help.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
               