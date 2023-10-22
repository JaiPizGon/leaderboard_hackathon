import streamlit as st
import gspread
from google.oauth2 import service_account

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
        ##  Connect with google sheets
        # Create a connection object.
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'],
        )
        gc = gspread.authorize(credentials)
        
        ## Read google sheet configuration
        google_sheets = st.secrets["google_sheets"]
        
        spreadsheet = gc.open(google_sheets['name_teams'])
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