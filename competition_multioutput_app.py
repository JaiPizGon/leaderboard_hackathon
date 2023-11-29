import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import gspread
from utils import read_users_file, validate_user
from datetime import datetime
import numpy as np
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

        validation_data = validation_data.map(convert_to_numeric)

    except gspread.exceptions.SpreadsheetNotFound:
        validation_data = None

    st.session_state.data_to_validate = validation_data
    st.session_state.n_outputs = validation_data.shape[1]
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


def calculate_error_metrics(predictions, true_values, output_name, capacity):
    """
    Calculate error metrics based on the problem type.

    Parameters:
    predictions (pd.Series or pd.DataFrame): Predicted values.
    true_values (pd.Series or pd.DataFrame): True values.
    output_name (str): name of the output
    capacity (float): maximum capacity of the source

    Returns:
    pd.DataFrame: A pandas DataFrame containing calculated error metrics and team information.
    """
    error_metrics = {}

    error_metrics[f"{output_name}_MAE"] = np.mean(np.abs(predictions - true_values))
    error_metrics[f"{output_name}_NV"] = predictions[predictions < 0].count()
    error_metrics[f"{output_name}_OV"] = predictions[predictions > capacity].count()
    error_metrics[f"{output_name}_score"] = (
        error_metrics[f"{output_name}_MAE"]
        + endesa["nv_penalization"] * error_metrics[f"{output_name}_NV"]
        + endesa["ov_penalization"] * error_metrics[f"{output_name}_OV"]
    )

    return error_metrics


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
        # If conversion fails, return the value itself
        print(f"Conversion failed for value: {s}")
        return s


def join_columns(row):
    integer_part = str(int(row[0]))  # Get the integer part and convert to string
    decimal_part = str(row[1])  # Get the decimal part
    return float(
        f"{integer_part}.{decimal_part}"
    )  # Join the parts and convert to float


def main():
    st.image(
        "https://github.com/JaiPizGon/leaderboard_hackathon/blob/master/resources/catedra/catedra-logo.png?raw=true",
        use_column_width=True,
    )
    # st.title(config["title"])

    # Sidebar
    st.sidebar.title("Upload your predictions")

    # User inputs
    team_name = st.sidebar.text_input("Team number")
    team_alias = st.sidebar.text_input("Team alias")
    password = st.sidebar.text_input("Password", type="password")
    uploaded_file = st.sidebar.file_uploader(
        "Choose predictions File", type=["csv", "dat"]
    )
    # commentary = st.sidebar.text_area("Brief description of your methodology")

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
            previous_metrics, _, previous_error_flag = read_previous_error_metrics()
            st.session_state.previous_metrics = previous_metrics
            if (team_alias != "") and (team_alias not in previous_metrics["Alias"]):
                # Read uploaded file
                if uploaded_file:
                    predictions = pd.read_csv(
                        uploaded_file, header=None, index_col=False, sep=";"
                    )

                    if predictions.shape[1] != st.session_state.n_outputs:
                        st.sidebar.error(
                            f"Error: predictions have {predictions.shape[1]} columns but {st.session_state.n_outputs} columns were expected"
                        )

                    elif (
                        predictions.shape[0]
                        != st.session_state.data_to_validate.shape[0]
                    ):
                        st.sidebar.error(
                            f"Error: predictions have {predictions.shape[0]} rows but {st.session_state.data_to_validate.shape[0]} rows were expected"
                        )

                    else:
                        # Convert all predictions to numeric
                        predictions = predictions.map(convert_to_numeric)

                        # Read validation data
                        if st.session_state.data_to_validate is None:
                            st.sidebar.error("Error: Validation data file not found.")

                        else:
                            # Check if team has not send more than n_tries
                            (
                                previous_metrics,
                                worksheet,
                                previous_error_flag,
                            ) = read_previous_error_metrics()
                            try:
                                team_tries = previous_metrics["Team"].value_counts()[
                                    team_name
                                ]
                            except KeyError:
                                team_tries = 0

                            if team_tries < config["n_tries"]:
                                error_metrics = {}
                                mae_score = 0
                                for o in range(st.session_state.n_outputs):
                                    o_error_metrics = calculate_error_metrics(
                                        predictions.iloc[:, o],
                                        st.session_state.data_to_validate.iloc[:, o],
                                        st.session_state.data_to_validate.columns[o],
                                        endesa["capacity"][o],
                                    )
                                    mae_score += o_error_metrics[
                                        f"{st.session_state.data_to_validate.columns[o]}_score"
                                    ]
                                    error_metrics.update(o_error_metrics)

                                # Calculate global score
                                error_metrics["Score"] = mae_score

                                # Create a DataFrame with calculated error metrics and team information
                                error_metrics["Team"] = team_name

                                # error_metrics["Commentary"] = commentary
                                error_metrics["Time"] = datetime.now().strftime(
                                    "%d/%m/%Y %H:%M:%S"
                                )

                                error_metrics["Alias"] = team_alias

                                error_metrics_df = pd.DataFrame([error_metrics])

                                # Update leaderboard and store error metrics
                                st.sidebar.write(
                                    "Updating leaderboard and storing error metrics..."
                                )

                                if previous_error_flag:
                                    st.sidebar.error(
                                        "Error: Leaderboard could not be loaded."
                                    )

                                else:
                                    try:
                                        worksheet.append_row(
                                            error_metrics_df.fillna(
                                                0.0
                                            ).values.tolist()[0]
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
                elif team_alias != "":
                    st.sidebar.error(
                        "Error: Choose a different alias, there is already a team with this one."
                    )
                else:
                    st.sidebar.error("Error: You must submit an alias.")
            else:
                st.sidebar.error("Error: File to be uploaded not found.")
        else:
            st.sidebar.error("Error: Incorrect user or password.")

    # Main area
    # st.title("Leaderboard")

    total_columns = sum(len(cols) + 3 for cols in endesa["col_show"])
    relative_widths = [(len(cols) + 3) / total_columns for cols in endesa["col_show"]]

    # Apply the CSS style
    leaderboard_columns = st.columns(
        # st.session_state.n_outputs + 1,
        spec=relative_widths,
        gap="medium",
    )

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
            st.session_state.previous_metrics,
            columns=["Team", "Alias"]
            + [item for sublist in endesa["col_show"] for item in sublist],
        )
        # Check if several rows for a single team must be displayed in leaderboard
        if not config["allow_duplicate_scores"]:
            # Sort the DataFrame by 'Score' column in ascending order
            prev_metrics.sort_values(by="Score", inplace=True)

            # Drop duplicates based on 'Team' column and keep the first occurrence (lowest 'Score' value)
            prev_metrics.drop_duplicates(subset="Team", keep="first", inplace=True)
        prev_metrics.iloc[:, 1:] = prev_metrics.iloc[:, 1:].map(convert_to_numeric)

        fs = 30
        # Define CSS styles
        th_props = [
            ("font-size", f"{fs}px"),  # Increase font size
            ("text-align", "center"),
            ("font-weight", "bold"),  # Make column headers bold
            # ("color", "#6d6d6d"),
            # ("background-color", "#f7ffff"),
        ]

        td_props = [("font-size", f"{fs}px")]  # Increase font size

        # Define a function to format values to three decimal places
        def format_float(val):
            return "{:.3f}".format(val)

        def format_integer(val):
            return "{:.0f}".format(val)

        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
            dict(
                selector=".row_heading", props=[("text-align", "left")]
            ),  # Align row headers to the left
        ]

        for i, cols in enumerate(endesa["col_show"]):
            try:
                out_name = cols[-1].split("_")[-1]
                leaderboard_columns[i].write(
                    f"<h2>{out_name}</h2>", unsafe_allow_html=True
                )  # Make the title bold
            except IndexError:
                leaderboard_columns[i].write(
                    f"<h2>Global Leaderboard</h2>", unsafe_allow_html=True
                )  # Make the title bold
            pm = prev_metrics[["Alias"] + cols]
            cols = [col.split("_")[0] for col in cols]
            pm.columns = ["Alias"] + cols

            # Define a function to truncate values to a maximum length
            def truncate_string(s, max_length):
                return s[:max_length]

            # Truncate values in the "Alias" column
            pm["Alias"] = pm["Alias"].apply(
                lambda x: truncate_string(x, config["max_alias_length"])
            )

            if pm.shape[1] > 2:
                pm["score"] = (
                    pm[cols[0]]
                    + endesa["nv_penalization"] * pm[cols[1]]
                    + endesa["ov_penalization"] * pm[cols[2]]
                )

                # Format the DataFrame values to three decimal places
                df_show = (
                    pm.sort_values(
                        by="score",
                        ascending=True,
                    )
                    .drop(columns=["score"])
                    .set_index("Alias")
                    .style.format(format_float, subset=cols[0])
                    .format(format_integer, subset=cols[1:])
                    .set_table_styles(styles)
                )
            else:
                df_show = (
                    pm.sort_values(
                        by=cols[0],
                        ascending=True,
                    )
                    .set_index("Alias")
                    .style.format(format_float, subset=cols[0])
                    .format(format_integer, subset=cols[1:])
                    .set_table_styles(styles)
                )
            leaderboard_columns[i].table(df_show)

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

    # Read endesa config
    endesa = st.secrets["endesa"]
    main()
