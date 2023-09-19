import streamlit as st
import pandas as pd
from arima_config import config_function
from google.oauth2 import service_account
import gspread
from datetime import datetime

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
        
    except gspread.exceptions.SpreadsheetNotFound:
        users_data = None
    
    st.session_state.users_data = users_data
    return 

def validate_user(team_name, password):
    """
    Validate user credentials.

    Parameters:
    team_name (str): The team name provided by the user.
    password (str): The password provided by the user.

    Returns:
    bool: True if user credentials are valid, False otherwise.
    """
    if team_name in st.session_state.users_data:
        if st.session_state.users_data[team_name] == password:
            return True
    return False

def read_previous_results():
    """
    Read previous results from Google Sheets using Google Sheets API.

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
        previous_results_df = pd.DataFrame(list_of_dicts)
        
        error_flag = 0
        
    except gspread.exceptions.SpreadsheetNotFound:
        previous_results_df = None
        worksheet = None
        error_flag = 1
    
    return previous_results_df, worksheet, error_flag


def read_series():
    """
    Read previous results from Google Sheets using Google Sheets API.

    Returns:
    tuple: A tuple containing two values:
        - pd.DataFrame: A pandas DataFrame containing the previous error metrics.
        - error_flag (int): An error flag indicating whether an error occurred during reading the file.
          - 0: No error occurred.
          - 1: File not found error occurred.
    """
    try:
        spreadsheet = gc.open(config_file['name_Arima'])
            
        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records(numericise_ignore=['all'])
        
        # Convert list of dicts to a DataFrame
        series_df = pd.DataFrame(list_of_dicts)
        
        series_df['weight'] = series_df['weight'].str.replace(',', '.').astype(float)
        
        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        previous_results_df = None

        error_flag = 1
    
    return series_df, error_flag


def main():
    # Set the main title based on the problem type from the config file
    st.title(f"{config_file['problem_type']} Hackathon")

    # Sidebar Section
    st.sidebar.title("User configuration")

    # Collect user inputs for team name and password
    team_name = st.sidebar.text_input("Team name")
    password = st.sidebar.text_input("Password", type="password")

    # Read series and check for errors
    if 'options' not in st.session_state:
        series, previous_error_flag = read_series()
        st.session_state.series = series
        if previous_error_flag:
            st.session_state.options = ()
            st.sidebar.error("Error: ARIMA series file could not be read.")
        else:
            st.session_state.options = [f"{str(r['Series'])} (reward: {str(r['weight'])})" for i, r in series.iterrows()]

    # Dropdown for selecting Series number
    option = st.sidebar.selectbox(
        'Series #',
        st.session_state.options
    )
    

    # Initialize session state for results if not already present
    if 'results' not in st.session_state:
        # Define columns based on the problem type
        if config_file['problem_type'] == 'ARMA':
            df_columns = ['Team', 'p', 'q']
        elif config_file['problem_type'] == 'ARIMA':
            df_columns = ['Team', 'p', 'd', 'q']
        elif config_file['problem_type'] == 'SARIMA':
            df_columns = ['Team', 'p', 'd', 'q', 'P', 'D', 'Q', 'f']

        st.session_state.results = pd.DataFrame(columns=df_columns)
    
    # Read team names and passwords
    if 'users_data' not in st.session_state:
        read_users_file()

    # Sidebar Buttons
    # Submit Button
    if st.sidebar.button("Submit"):
        # Handling code after clicking submit
        # Validate user
        if validate_user(team_name, password):
            # Update leaderboard and store error metrics
            st.sidebar.write("Storing results...")
            previous_results, worksheet, previous_error_flag = read_previous_results()
            
            # Check if team has not send more than n_tries
            try:
                team_tries = previous_results.loc[previous_results['Series'] == option.split(' ')[0], 'Team'].value_counts()[team_name]
            except KeyError:
                team_tries = 0
            
            if team_tries < config_file['n_tries']:
                # Read result from the student
                entry_empty = False
                df_dict = {'Team': team_name, 'Series': option.split(' ')[0]}
                for index, col in enumerate(st.session_state.results.columns[1:]):
                    df_dict[col] = st.session_state[col]
                    if st.session_state[col] == '':
                        entry_empty = True
                        break
                df_dict['Time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    
                if entry_empty:
                    st.sidebar.error("Error: could not submit empty inputs")
                    
                else:
                    result_sent = pd.DataFrame(df_dict, index=[0])
                    
                    if previous_error_flag:
                        st.sidebar.error("Error: Previous tries could not be loaded.")
                    
                    else:
                        try:
                            worksheet.append_row(result_sent.values.tolist()[0])
                            st.sidebar.success(f"Results updated. You have {config_file['n_tries'] - team_tries - 1} tries left.")
                        except:
                            st.sidebar.error("Error: Leaderboard could not be updated.")
            else:
                st.sidebar.error(f"Error: {team_name} has exceeded the number of tries for Serie {option}.")
        else:
            st.sidebar.error("Error: Incorrect user or password.")
            
    # Main area
    st.title("Input parameters")

    # Text inputs for each parameter
    txtColumns = st.columns(len(st.session_state.results.columns) - 1)
    for index, col in enumerate(st.session_state.results.columns[1:]):
        with txtColumns[index]:
            st.text_input(col, key=col)
        

if __name__ == "__main__":

    ##  Connect with google sheets
    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'],
    )
    gc = gspread.authorize(credentials)
    
    main()
