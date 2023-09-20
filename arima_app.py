import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import gspread
from datetime import datetime
import bar_chart_race as bcr
import base64

# Make the web page fill the full area
st.set_page_config(layout="wide")


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

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
        spreadsheet = gc.open(google_sheets['name_arima'])
            
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
        spreadsheet = gc.open(f"{google_sheets['name_arima']}_Series")
            
        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records(numericise_ignore=['all'])
        
        # Convert list of dicts to a DataFrame
        series_df = pd.DataFrame(list_of_dicts)
        
        series_df['weight'] = series_df['weight'].str.replace(',', '.').astype(float)
        
        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        series_df = None

        error_flag = 1
    
    return series_df, error_flag

# Function to clear the session state for showing solutions
def clear_show_solution():
    st.session_state.show_solution = False
    st.session_state.show_results = False
    return


def main():
    # Set the main title based on the problem type from the config file
    st.title(f"{config['arima_problem_type']} Hackathon")

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
        st.session_state.options,
        on_change=clear_show_solution
    )

    # Initialize session state for results if not already present
    if 'results' not in st.session_state:
        # Define columns based on the problem type
        if config['arima_problem_type'] == 'ARMA':
            df_columns = ['Team', 'p', 'q']
        elif config['arima_problem_type'] == 'ARIMA':
            df_columns = ['Team', 'p', 'd', 'q']
        elif config['arima_problem_type'] == 'SARIMA':
            df_columns = ['Team', 'p', 'd', 'q', 'P', 'D', 'Q', 'f']

        st.session_state.results = pd.DataFrame(columns=df_columns)

    # Initialize session state for showing solution
    if 'show_solution' not in st.session_state:
        st.session_state.show_solution = False
    
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # Solution area
    # Button for showing solution
    if st.sidebar.button("Show solution"):
        # Admin check for showing solution
        if team_name == admin_account['name'] and password == admin_account['password']:
            st.session_state.show_solution = True
        else:
            st.session_state.show_solution = False
            st.sidebar.error("Error: wrong admin name or password")
    
    # Button for showing results
    if st.sidebar.button('Show results'):
        # Admin check for showing results
        if team_name == admin_account['name'] and password == admin_account['password']:
            st.session_state.show_results = True
        else:
            st.session_state.show_results = False
            st.sidebar.error("Error: wrong admin name or password")
            
    # Main area
    st.title("Input parameters")

    # Text inputs for each parameter
    txtColumns = st.columns(len(st.session_state.results.columns) - 1)
    for index, col in enumerate(st.session_state.results.columns[1:]):
        with txtColumns[index]:
            st.text_input(col, key=col)
        
    
    if st.session_state.show_solution:
        series_solution = series.loc[series['Series'] == option.split(' ')[0], st.session_state.results.columns[1:]]
        new_index = ['Series #' + str(x + 1) for x in series_solution.index.values]
        series_solution.index = new_index
        st.dataframe(pd.DataFrame(series_solution))
    
    if st.session_state.show_results:
        previous_results, _, _ = read_previous_results()
        
        # Convert time to datetime
        previous_results['Time'] = pd.to_datetime(previous_results['Time'])
        
        # Merge previous results with the series DataFrame on specific columns
        # This combines the 'Series', 'p', and 'q' columns from both DataFrames
        merged_df = pd.merge(previous_results.reset_index(), st.session_state.series, 
                            on=['Series', 'p', 'q'], how='left')

        # Fill any NaN values in the 'weight' column with zeros
        # This ensures that all teams have a 'weight' even if they haven't made any attempts
        merged_df['weight'].fillna(0, inplace=True)

        # Create a 'mark' column to hold the cumulative sum of 'weight' for each team
        # This will be used for scoring, and is multiplied by 100 for percentage representation
        merged_df['mark'] = merged_df.groupby('Team')['weight'].cumsum() * 100

        # Re-set the index of the merged DataFrame to 'Time'
        # This is to ensure that the DataFrame is indexed by time, which is likely important for time-series data
        merged_df.set_index('Time', inplace=True)

        # Number of steps for expanding the DataFrame index
        n_steps = config['n_steps']

        # Filter the DataFrame to include only 'Team' and 'mark' columns
        df = merged_df[['Team', 'mark']]

        # Sort by 'mark' in descending and 'Time' in ascending order
        sorted_df = merged_df.sort_values(by=['mark', 'Time'], ascending=[False, True])

        # Get the team with the highest mark that reached it the fastest
        winning_team = sorted_df.iloc[0]['Team']
        
        # Pivot the DataFrame so that each 'Team' becomes a column
        df = df.pivot(columns='Team', values='mark')
        
        # Create bar chart race animation
        try:
            html_str = bcr.bar_chart_race(
                    df.ffill().fillna(0), 
                    interpolate_period=True, steps_per_period=n_steps, title='Team leaderboard evolution',
                    period_fmt='%H:%m'
                ).data
        except AttributeError:
            html_str = bcr.bar_chart_race(
                    df.ffill().fillna(0), 
                    interpolate_period=True, steps_per_period=n_steps, title='Team leaderboard evolution',
                    period_fmt='%H:%m'
                )
            
        start = html_str.find('base64,') + len('base64,')
        end = html_str.find('">')

        video = base64.b64decode(html_str[start:end])
        
        st.video(video)
        
        # Show winning team name
        st.subheader(f"Winning team: {winning_team}")

        # Filtering the DataFrame to include only rows with max 'mark' for each 'Team'
        filtered_df = sorted_df.loc[sorted_df.groupby('Team')['mark'].idxmax()]
        
        filtered_df = filtered_df[['Team','mark']]

        csv = convert_df(filtered_df)

        st.download_button(
            label="Download marks as CSV",
            data=csv,
            file_name='marks.csv',
            mime='text/csv',
        )

        

if __name__ == "__main__":

    ##  Connect with google sheets
    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'],
    )
    gc = gspread.authorize(credentials)
    
    ## Read Admin account configuration
    admin_account = st.secrets["admin_account"]
    
    ## Read google sheet configuration
    google_sheets = st.secrets["google_sheets"]
    
    ## Read config 
    config = st.secrets["config"]
    main()
