import streamlit as st
import pickle
import pandas as pd

# Load the trained Linear Regression model
with open('LinearRegression.pkl', 'rb') as model_file:
    linreg_model = pickle.load(model_file)

sales = pd.read_csv('advertising.csv')
X = sales.drop('Sales', axis=1)

# Streamlit UI
st.title('Sales Prediction App')

# Sidebar with input fields
st.sidebar.header('Input Amount Spent on Advert')

tv_ad = st.sidebar.number_input('How much was spent on TV Advertising ($)', value=float(X['TV'].mean()))
radio_ad = st.sidebar.number_input('How much was spent on Radio Advertising ($)', value=float(X['Radio'].mean()))
newspaper_ad = st.sidebar.number_input('How much was spent on Newspaper Advertising ($)', value=float(X['Newspaper'].mean()))

# Display the input ad spending
st.sidebar.markdown('**Amount spent on Advertising:**')
st.sidebar.write(f'TV: ${tv_ad:.2f}')
st.sidebar.write(f'Radio: ${radio_ad:.2f}')
st.sidebar.write(f'Newspaper: ${newspaper_ad:.2f}')

# Predict button
if st.sidebar.button('Predict'):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'TV': [tv_ad],
        'Radio': [radio_ad],
        'Newspaper': [newspaper_ad]
    })

    # Predict sales
    predicted_sales = linreg_model.predict(input_data)[0]

    # Display the predicted sales
    st.write(f'Predicted Sales: ${predicted_sales:.2f}')
