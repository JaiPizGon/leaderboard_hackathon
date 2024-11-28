import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import gspread
from utils import read_users_file, validate_user
from datetime import datetime
from sklearn.metrics import confusion_matrix
from datetime import datetime
import numpy as np
from statsmodels.tsa.stattools import acf
import time

# Make the web page fill the full area
st.set_page_config(layout="wide")


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
        spreadsheet = gc.open(config["leaderboard_validation_file"])

        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records(numericise_ignore=["all"])

        # Convert list of dicts to a DataFrame
        validation_data = pd.DataFrame(list_of_dicts)

        if config["leaderboard_problem_type"] != "classification":
            validation_data = validation_data.map(convert_to_numeric)

    except gspread.exceptions.SpreadsheetNotFound:
        validation_data = None

    st.session_state.data_to_validate = validation_data
    return


def correct_name(data):
    # Get the name of the column
    column_name = data.columns[0]

    # Create a new DataFrame with the column name as the next row
    column_name_df = pd.DataFrame({column_name: [column_name]})

    # Concatenate the two DataFrames vertically
    result_df = pd.concat([data, column_name_df], ignore_index=True)

    # Rename column
    result_df = result_df.rename(columns={column_name: "value"})
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

    if config["leaderboard_problem_type"] == "regression":
        residuals = predictions.values - true_values.values
        # RMSE (Root Mean Squared Error)
        error_metrics["RMSE"] = np.sqrt(np.mean((residuals) ** 2))

        # Calculate MAPE (Mean Absolute Percentage Error)
        error_metrics["MAPE"] = np.mean(np.abs((residuals) / true_values)) * 100

        # Calculate negative_values (count of negative predictions)
        negative_values = (predictions < 0).sum()[0]
        error_metrics["Negative_Values"] = negative_values

    elif config["leaderboard_problem_type"] == "classification":
        if predictions.shape[0] == true_values.shape[0] - 1:
            predictions = correct_name(predictions)
        if predictions.shape[0] == true_values.shape[0] + 1:
            predictions = predictions.iloc[1:]

        ## Obtain confusion matrix
        cm = confusion_matrix(true_values, predictions.astype(type(true_values.values[0,0])))

        # Total of predictions
        t = predictions.shape[0]

        # Accuracy
        error_metrics["Accuracy"] = (cm[0, 0] + cm[1, 1]) / t

        # Calculate sensitivity (True Positive Rate)
        error_metrics["Sensitivity"] = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        # Calculate specificity (True Negative Rate)
        error_metrics["Specificity"] = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    elif config["leaderboard_problem_type"] == "forecast":
        residuals = predictions.values - true_values.values
        # ME (Mean Error)
        error_metrics["ME"] = np.mean(residuals)

        # RMSE (Root Mean Squared Error)
        error_metrics["RMSE"] = np.sqrt(np.mean((residuals) ** 2))

        # MAE (Mean Absolute Error)
        error_metrics["MAE"] = np.mean(np.abs(residuals))

        # MPE (Mean Percentage Error)
        error_metrics["MPE"] = np.mean((residuals) / true_values) * 100

        # MAPE (Mean Absolute Percentage Error)
        error_metrics["MAPE"] = np.mean(np.abs((residuals) / true_values)) * 100

        # ACF1 (Auto-correlation Function at lag 1)
        # Note: It's applied on the residuals
        error_metrics["ACF1"] = acf(residuals, nlags=1)[1]

        # Theil's U statistic
        numerator = np.sqrt(np.mean((residuals) ** 2))
        denominator = np.sqrt(np.mean(predictions**2) + np.mean(true_values**2))
        error_metrics["Theils'U"] = numerator / denominator
    elif config["leaderboard_problem_type"] == "icade":
        residuals = predictions.values - true_values.values

        # RMSE (Root Mean Squared Error)
        error_metrics["RMSE"] = np.sqrt(np.mean((residuals) ** 2))

        # MAE (Mean Absolute Error)
        error_metrics["MAE"] = np.mean(np.abs(residuals))

        # MAPE (Mean Absolute Percentage Error)
        error_metrics["MAPE"] = np.mean(np.abs((residuals) / true_values)) * 100
    # Create a DataFrame with calculated error metrics and team information
    error_metrics["Team"] = team_name
    error_metrics["Commentary"] = commentary
    error_metrics["Time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
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
        spreadsheet = gc.open(google_sheets["name_leaderboard"])

        worksheet = spreadsheet.get_worksheet(0)
        list_of_dicts = worksheet.get_all_records(numericise_ignore=["all"])

        # Convert list of dicts to a DataFrame
        previous_metrics_df = pd.DataFrame(list_of_dicts)

        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        previous_metrics_df = None
        worksheet = None
        error_flag = 1

    return previous_metrics_df, worksheet, error_flag


def convert_to_numeric(s):
    if isinstance(s, (float, int)):
        return s  # Return s as is if it's already a float or int
    try:
        # Replace commas with periods, then try to convert to a float
        return float(s.replace(",", "."))
    except ValueError:
        # If conversion fails, return None
        # print(f"Conversion failed for value: {s}")
        return None


def join_columns(row):
    integer_part = str(int(row[0]))  # Get the integer part and convert to string
    decimal_part = str(row[1])  # Get the decimal part
    return float(
        f"{integer_part}.{decimal_part}"
    )  # Join the parts and convert to float


def main():
    st.title(
        f"{config['title']}"
    )

    # Sidebar
    st.sidebar.title("Upload your predictions")

    # User inputs
    team_name = st.sidebar.text_input("Team name")
    password = st.sidebar.text_input("Password", type="password")
    uploaded_file = st.sidebar.file_uploader(
        "Choose predictions File", type=["csv", "dat"]
    )
    commentary = st.sidebar.text_area("Brief description of your methodology")

    # Obtain if it is admin account
    admin_enabled = (
        team_name == admin_account["name"] and password == admin_account["password"]
    )

    # Enable auto refresh
    auto_refresh = st.sidebar.checkbox(
        "Enable auto-refresh", disabled=not admin_enabled
    )
    refresh_interval = st.sidebar.number_input(
        "Refresh interval (seconds)", min_value=1, value=30, disabled=not admin_enabled
    )

    # Read team names and passwords
    if "users_data" not in st.session_state:
        read_users_file()

    # Read data to validate
    if "data_to_validate" not in st.session_state:
        read_data_to_validate()

    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = datetime.now()

    if "not_shown" not in st.session_state:
        st.session_state.not_shown = True

    # Buttons
    if st.sidebar.button("Submit"):
        # Handling code after clicking submit
        # Validate user
        if validate_user(team_name, password):
            # Read uploaded file
            if uploaded_file:
                predictions = pd.read_csv(uploaded_file, header=None, index_col=False)

                if predictions.shape[1] > 1:
                    predictions = pd.DataFrame(
                        predictions.apply(join_columns, axis=1), columns=["prediction"]
                    )
                # Read validation data
                if st.session_state.data_to_validate is None:
                    st.sidebar.error("Error: Validation data file not found.")

                else:
                    error_metrics = calculate_error_metrics(
                        predictions,
                        st.session_state.data_to_validate,
                        team_name,
                        commentary,
                    )

                    # Update leaderboard and store error metrics
                    st.sidebar.write(
                        "Updating leaderboard and storing error metrics..."
                    )
                    (
                        previous_metrics,
                        worksheet,
                        previous_error_flag,
                    ) = read_previous_error_metrics()

                    # Check if team has not send more than n_tries
                    try:
                        team_tries = previous_metrics["Team"].value_counts()[team_name]
                    except KeyError:
                        team_tries = 0

                    if team_tries < config["n_tries"]:
                        if previous_error_flag:
                            st.sidebar.error("Error: Leaderboard could not be loaded.")
                        else:
                            try:
                                worksheet.append_row(
                                    error_metrics[previous_metrics.columns if previous_metrics.columns.any() else config["col_show"] + [x for x in error_metrics.columns if x not in config["col_show"]]].fillna(0.0).values.tolist()[0]
                                )
                                st.sidebar.success(
                                    f"Leaderboard updated. You have {config['n_tries'] - team_tries - 1} tries left."
                                )
                                st.session_state.not_shown = True
                            except:
                                st.sidebar.error(
                                    "Error: Leaderboard could not be updated."
                                )
                    else:
                        st.sidebar.error(
                            f"Error: team has exceeded the number of tries ({config['n_tries']})"
                        )
            else:
                st.sidebar.error("Error: File to be uploaded not found.")
        else:
            st.sidebar.error("Error: Incorrect user or password.")

    # Main area
    st.title("Leaderboard")

    leaderboard_placeholder = st.empty()
    if st.sidebar.button("Refresh Leaderboard") or auto_refresh:
        seconds_passed = (
            datetime.now() - st.session_state.last_refresh_time
        ).total_seconds()
        submit_time_enabled = (seconds_passed / 60 > config["n_minutes"]) or (
            auto_refresh and (seconds_passed > refresh_interval)
        )

        if submit_time_enabled or admin_enabled or st.session_state.not_shown:
            # Update last refresh time
            st.session_state.last_refresh_time = datetime.now()

            # Refresh leaderboard and display
            st.sidebar.write("Refreshing leaderboard...")
            previous_metrics, _, previous_error_flag = read_previous_error_metrics()
            st.session_state.previous_metrics = previous_metrics
            st.session_state.not_shown = False
            st.sidebar.write("Leaderboard refreshed.")
            if previous_error_flag:
                st.sidebar.error("Error: Leaderboard could not be loaded.")
        else:
            st.sidebar.warning(
                f"You must wait {config['n_minutes'] * 60 - int(seconds_passed)} seconds to refresh again."
            )
    
    if "previous_metrics" in st.session_state:
        prev_metrics = pd.DataFrame(
            st.session_state.previous_metrics, columns=["Team"] + config["col_show"]
        )
        prev_metrics.iloc[:, 1:] = prev_metrics.iloc[:, 1:].map(convert_to_numeric)
        df = prev_metrics.sort_values(
                by=config["col_show"][0],
                ascending=not (config["leaderboard_problem_type"] == "classification"),
            ).reset_index(drop=True)
        leaderboard_placeholder.table(
            df.style.set_table_styles([{'selector': 'td, th', 'props': [('font-size', f'{config["font_size"]}px')]}]).hide(axis="index").format(precision=2),
        )
            # use_container_width=False,
            # hide_index=True,
            # height=config["leaderboard_height"]  # Adjust the height of the container (in pixels)

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


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

    # Read admin account configuration
    admin_account = st.secrets["admin_account"]

    ## Read google sheet configuration
    google_sheets = st.secrets["google_sheets"]

    # Read config
    config = st.secrets["config"]

    main()
