import streamlit as st
import pandas as pd
from config import config_function
from google.oauth2 import service_account
import gspread

# Make the web page fill the full area
st.set_page_config(layout="wide")

# Load config from config.py
config_file = config_function()

def read_users_file():
    """
    Read user data from Google Sheets using Google Sheets API.

    Returns:
    tuple: A tuple containing two values:
        - users_data (dict or None): A dictionary containing user data with team names as keys and passwords as values,
          or None if the spreadsheet or worksheet could not be found.
        - error_flag (int): An error flag indicating whether an error occurred during accessing the spreadsheet.
          - 0: No error occurred.
          - 1: Spreadsheet not found error occurred.
    """
    try:
        spreadsheet = gc.open(config_file['pwd_table'])
        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records()
        
        # Convert list of dicts to a dictionary
        users_data = {record['Team']: record['Password'] for record in list_of_dicts}
        
        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        users_data = None
        error_flag = 1
        
    return users_data, error_flag

def validate_user(users_data, team_name, password):
    """
    Validate user credentials.

    Parameters:
    users_data (dict): A dictionary containing user data with team names as keys and hashed passwords as values.
    team_name (str): The team name provided by the user.
    password (str): The password provided by the user.

    Returns:
    bool: True if user credentials are valid, False otherwise.
    """
    if team_name in users_data:
        if users_data[team_name] == password:
            return True
    return False

def read_data_to_validate():
    """
    Read and validate prediction data from the provided data file.

    Returns:
    tuple: A tuple containing two values:
        - pd.DataFrame: A pandas DataFrame containing the validation data.
        - error_flag (int): An error flag indicating whether an error occurred during reading the file.
          - 0: No error occurred.
          - 1: File not found error occurred.
    
    """
    try:
        spreadsheet = gc.open(config_file['type_problem'])
            
        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records()
        
        # Convert list of dicts to a DataFrame
        validation_data = pd.DataFrame(list_of_dicts)
        
        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        validation_data = None
        error_flag = 1
        
    return validation_data, error_flag

def correct_name(data):
    # Get the name of the column
    column_name = data.columns[0]

    # Create a new DataFrame with the column name as the next row
    column_name_df = pd.DataFrame({column_name: [column_name]})

    # Concatenate the two DataFrames vertically
    result_df = pd.concat([data, column_name_df], ignore_index=True)
    
    # Rename column
    result_df = result_df.rename(columns={column_name: 'value'})
    return result_df

def calculate_error_metrics(predictions, true_values, team_name, commentary):
    """
    Calculate error metrics based on the problem type.

    Parameters:
    predictions (pd.Series or pd.DataFrame): Predicted values.
    true_values (pd.Series or pd.DataFrame): True values.
    team_name (str): Team name.
    commentary (str): Commentary on methodology.

    Returns:
    pd.DataFrame: A pandas DataFrame containing calculated error metrics and team information.
    """
    error_metrics = {}

    predictions = correct_name(predictions)
    true_values = correct_name(true_values)
    
    if config_file['type_problem'] == 'regression':
        # Calculate RMSE (Root Mean Squared Error)
        rmse = ((predictions - true_values) ** 2).mean() ** 0.5
        error_metrics['RMSE'] = rmse

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = ((abs(predictions - true_values) / true_values) * 100).mean()
        error_metrics['MAPE'] = mape

        # Calculate negative_values (count of negative predictions)
        negative_values = (predictions < 0).sum()
        error_metrics['Negative_Values'] = negative_values
    elif config_file['type_problem'] == 'classification':
        # Assuming 'predictions' and 'true_values' are binary (0 or 1) for classification
        correct_predictions = (predictions == true_values)
        accuracy = correct_predictions.mean()
        error_metrics['Accuracy'] = accuracy[0]

        # Calculate sensitivity (True Positive Rate)
        true_positive = correct_predictions.sum()
        sensitivity = true_positive / len(true_values[true_values == 1])
        error_metrics['Sensitivity'] = sensitivity[0]

        # Calculate specificity (True Negative Rate)
        true_negative = len(predictions[correct_predictions]) - true_positive
        specificity = true_negative / len(true_values[true_values == 0])
        error_metrics['Specificity'] = specificity[0]

    # Create a DataFrame with calculated error metrics and team information
    error_metrics['Team'] = team_name
    error_metrics['Commentary'] = commentary
    error_metrics_df = pd.DataFrame([error_metrics])

    return error_metrics_df


def read_previous_error_metrics():
    """
    Read previous error metrics from Google Sheets using Google Sheets API.

    Returns:
    tuple: A tuple containing two values:
        - pd.DataFrame: A pandas DataFrame containing the previous error metrics.
        - error_flag (int): An error flag indicating whether an error occurred during reading the file.
          - 0: No error occurred.
          - 1: File not found error occurred.
    """
    try:
        spreadsheet = gc.open(config_file['name_GDrive'])
            
        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records(numericise_ignore=['all'])
        
        # Convert list of dicts to a DataFrame
        previous_metrics_df = pd.DataFrame(list_of_dicts)
        
        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        previous_metrics_df = None
        worksheet = None
        error_flag = 1
    
    return previous_metrics_df, worksheet, error_flag


def main():
    st.title(config_file['title'])
    
    # Sidebar
    st.sidebar.title("Upload your predictions")
    
    # User inputs
    team_name = st.sidebar.text_input("Team name")
    password = st.sidebar.text_input("Password", type="password")
    uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=['csv'])
    commentary = st.sidebar.text_area("Brief description of your methodology")
    
    # Buttons
    if st.sidebar.button("Submit"):
        
        # Read user data:
        users_data, users_data_error = read_users_file()
        if users_data_error:
            st.sidebar.error("Error: users.json file could not be read.")

        else:
            # Validate user
            if validate_user(users_data, team_name, password):

                # Read uploaded file
                if uploaded_file:
                    predictions = pd.read_csv(uploaded_file)
                    
                    # Read validation data
                    data_to_validate, val_data_error = read_data_to_validate()
                    if val_data_error:
                        st.sidebar.error("Error: Validation data file not found.")
                        
                    else:
                        error_metrics = calculate_error_metrics(predictions, data_to_validate, team_name, commentary)
                    
                    # Update leaderboard and store error metrics
                    st.sidebar.write("Updating leaderboard and storing error metrics...")
                    previous_metrics, worksheet, previous_error_flag = read_previous_error_metrics()
                    
                    # Check if team has not send more than n_tries
                    try:
                        team_tries = previous_metrics['Team'].value_counts()[team_name]
                    except KeyError:
                        team_tries = 0
                        
                    if team_tries < config_file['n_tries']:
                        if previous_error_flag:
                            st.sidebar.error("Error: Leaderboard could not be loaded.")
                        else:
                            try:
                                worksheet.append_row(error_metrics.values.tolist()[0])
                                st.sidebar.success(f"Leaderboard updated. You have {config_file['n_tries'] - team_tries - 1} tries left.")
                            except:
                                st.sidebar.error("Error: Leaderboard could not be updated.")
                    else:
                        st.sidebar.error(f"Error: team has exceeded the number of tries ({config_file['n_tries']})")
                else:
                    st.sidebar.error("Error: File to be uploaded not found.")
            else:
                st.sidebar.error("Error: Incorrect user or password.")
    
    if st.sidebar.button("Refresh Leaderboard"):
        # Refresh leaderboard and display
        st.sidebar.write("Refreshing leaderboard...")
        previous_metrics, _, previous_error_flag = read_previous_error_metrics()
        if previous_error_flag:
            st.sidebar.error("Error: Leaderboard could not be loaded.")
    
    # Main area
    st.title("Leaderboard")
    try:
        st.dataframe(pd.DataFrame(previous_metrics, columns=['Team'] + config_file['col_show']).sort_values(by=config_file['col_show'][0], ascending=False))
    except UnboundLocalError:
        previous_metrics, _, previous_error_flag = read_previous_error_metrics()
        if previous_error_flag:
            st.sidebar.error("Error: Leaderboard could not be loaded.")
        else:
            st.dataframe(pd.DataFrame(previous_metrics, columns=['Team'] + config_file['col_show']).sort_values(by=config_file['col_show'][0], ascending=False))
            st.sidebar.write("Leaderboard refreshed.")
        

if __name__ == "__main__":

    ##  Connect with google sheets
    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'],
    )
    gc = gspread.authorize(credentials)
    
    main()
