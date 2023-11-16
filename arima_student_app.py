import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import gspread
from datetime import datetime
from utils import read_users_file, validate_user


# Make the web page fill the full area
st.set_page_config(layout="wide")


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
        spreadsheet = gc.open(google_sheets["name_arima"])

        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records(numericise_ignore=["all"])

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
        list_of_dicts = worksheet.get_all_records(numericise_ignore=["all"])

        # Convert list of dicts to a DataFrame
        series_df = pd.DataFrame(list_of_dicts)

        series_df["weight"] = series_df["weight"].str.replace(",", ".").astype(float)

        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        series_df = None

        error_flag = 1

    return series_df, error_flag


def main():
    # Set the main title based on the problem type from the config file
    st.title(f"{config['arima_problem_type']} Hackathon")

    # Sidebar Section
    st.sidebar.title("User configuration")

    # Collect user inputs for team name and password
    team_name = st.sidebar.text_input("Team name")
    password = st.sidebar.text_input("Password", type="password")

    # Read series and check for errors
    if "options" not in st.session_state:
        series, previous_error_flag = read_series()
        st.session_state.series = series
        if previous_error_flag:
            st.session_state.options = ()
            st.sidebar.error("Error: ARIMA series file could not be read.")
        else:
            st.session_state.options = [
                f"{str(r['Series'])} (reward: {str(r['weight'])})"
                for i, r in series.iterrows()
            ]

    # Dropdown for selecting Series number
    option = st.sidebar.selectbox("Series #", st.session_state.options)

    # Initialize session state for results if not already present
    if "results" not in st.session_state:
        # Define columns based on the problem type
        if config["arima_problem_type"] == "ARMA":
            df_columns = ["Team", "p", "q", "include_mean", "lambda"]
        elif config["arima_problem_type"] == "ARIMA":
            df_columns = ["Team", "p", "d", "q", "include_mean", "lambda"]
        elif config["arima_problem_type"] == "SARIMA":
            df_columns = [
                "Team",
                "p",
                "d",
                "q",
                "P",
                "D",
                "Q",
                "s",
                "include_mean",
                "lambda",
            ]

        st.session_state.results = pd.DataFrame(columns=df_columns)

    # Read team names and passwords
    if "users_data" not in st.session_state:
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
                team_tries = previous_results.loc[
                    previous_results["Series"] == option.split(" ")[0], "Team"
                ].value_counts()[team_name]
            except KeyError:
                team_tries = 0

            if team_tries < config["n_tries"]:
                # Read result from the student
                entry_empty = False
                df_dict = {"Team": team_name, "Series": option.split(" ")[0]}
                for index, col in enumerate(st.session_state.results.columns[1:]):
                    if col == "include_mean":
                        df_dict[col] = int(st.session_state[col])
                    else:
                        df_dict[col] = st.session_state[col]
                    if st.session_state[col] == "":
                        entry_empty = True
                        break
                df_dict["Time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                if entry_empty:
                    st.sidebar.error("Error: could not submit empty inputs")

                else:
                    result_sent = pd.DataFrame(df_dict, index=[0])

                    if previous_error_flag:
                        st.sidebar.error("Error: Previous tries could not be loaded.")

                    else:
                        try:
                            worksheet.append_row(result_sent.values.tolist()[0])
                            st.sidebar.success(
                                f"Results updated. You have {config['n_tries'] - team_tries - 1} tries left."
                            )
                        except:
                            st.sidebar.error("Error: Leaderboard could not be updated.")
            else:
                st.sidebar.error(
                    f"Error: {team_name} has exceeded the number of tries for Serie {option}."
                )
        else:
            st.sidebar.error("Error: Incorrect user or password.")

    # Main area
    st.title("Input parameters")

    # Text inputs for each parameter
    txtColumns = st.columns(len(st.session_state.results.columns) - 1)
    for index, col in enumerate(st.session_state.results.columns[1:-2]):
        with txtColumns[index]:
            st.text_input(col, key=col)
    with txtColumns[index + 1]:
        st.text_input(
            st.session_state.results.columns[-1],
            key=st.session_state.results.columns[-1],
        )
    st.checkbox(
        st.session_state.results.columns[-2], key=st.session_state.results.columns[-2]
    )


if __name__ == "__main__":
    ##  Connect with google sheets
    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    gc = gspread.authorize(credentials)

    ## Read google sheet configuration
    google_sheets = st.secrets["google_sheets"]

    ## Read config
    config = st.secrets["config"]

    main()
