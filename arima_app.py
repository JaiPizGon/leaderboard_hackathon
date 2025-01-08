import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import gspread
import matplotlib.pyplot as plt
import bar_chart_race as bcr
import base64
import numpy as np
from datetime import datetime
import time

# Make the web page fill the full area
st.set_page_config(layout="wide")


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


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
        if previous_results_df.shape[0] != 0:
            previous_results_df["lambda"] = (
                previous_results_df["lambda"].str.replace(",", ".").astype(float)
            )
        previous_results_df = previous_results_df.drop_duplicates(subset=previous_results_df.columns[previous_results_df.columns != 'Time'], keep='first')
        previous_results_df.reset_index(drop=True, inplace=True)
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
        series_df["lambda"] = series_df["lambda"].str.replace(",", ".").astype(float)

        error_flag = 0
    except gspread.exceptions.SpreadsheetNotFound:
        series_df = None

        error_flag = 1

    return series_df, error_flag


# Function to clear the session state for showing solutions
def clear_show_solution():
    st.session_state.show_solution = False
    st.session_state.show_results = False
    st.session_state.show_leaderboard = False
    return


def main():
    # Set the main title based on the problem type from the config file
    st.title(f"{config['arima_problem_type']} Hackathon")

    # Sidebar Section
    st.sidebar.title("User configuration")

    # Collect user inputs for team name and password
    team_name = st.sidebar.text_input("Team name")
    password = st.sidebar.text_input("Password", type="password")

    # Obtain if it is admin account
    admin_enabled = (
        team_name == admin_account["name"] and password == admin_account["password"]
    )

    # Read series and check for errors
    if "options" not in st.session_state:
        series, previous_error_flag = read_series()
        st.session_state.series = series
        if previous_error_flag:
            st.session_state.options = ()
            st.sidebar.error("Error: ARIMA series file could not be read.")
        else:
            st.session_state.options = [
                f"{str(r['Series'])} (reward: {str(r['weight'] * 100)})"
                for _, r in series.drop_duplicates(subset='Series', keep=False).iterrows()
            ]

    # Dropdown for selecting Series number
    option = st.sidebar.selectbox(
        "Series #", st.session_state.options, on_change=clear_show_solution
    )

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

    # Initialize session state for showing solution
    if "show_solution" not in st.session_state:
        st.session_state.show_solution = False

    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    if "show_leaderboard" not in st.session_state:
        st.session_state.show_leaderboard = False

    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = datetime.now()

    # Solution area
    # Button for showing solution
    if st.sidebar.button("Show solution"):
        # Admin check for showing solution
        if team_name == admin_account["name"] and password == admin_account["password"]:
            st.session_state.show_solution = True
        else:
            st.session_state.show_solution = False
            st.sidebar.error("Error: wrong admin name or password")

    # Button for showing results
    if st.sidebar.button("Show leaderboard (dynamic)"):
        # Admin check for showing results
        if team_name == admin_account["name"] and password == admin_account["password"]:
            st.session_state.show_results = True
        else:
            st.session_state.show_results = False
            st.sidebar.error("Error: wrong admin name or password")

    # Button for showing results
    if st.sidebar.button("Show leaderboard (static)"):
        # Admin check for showing results
        if team_name == admin_account["name"] and password == admin_account["password"]:
            st.session_state.show_leaderboard = True
        else:
            st.session_state.show_leaderboard = False
            st.sidebar.error("Error: wrong admin name or password")

    # Enable auto refresh
    auto_refresh = st.sidebar.checkbox(
        "Enable auto-refresh", disabled=not admin_enabled
    )

    # Only last try
    only_last_try = st.sidebar.checkbox(
        "Only last try", disabled=not admin_enabled, value=True
    )

    refresh_interval = st.sidebar.number_input(
        "Refresh interval (seconds)", min_value=1, value=30, disabled=not admin_enabled
    )


    if st.session_state.show_solution:
        series_solution = st.session_state.series.loc[
            st.session_state.series["Series"] == option.split(" ")[0],
            st.session_state.results.columns[1:],
        ]
        new_index = ["Series #" + str(x + 1) for x in series_solution.index.values]
        series_solution.index = new_index
        st.dataframe(pd.DataFrame(series_solution))

    seconds_passed = (
        datetime.now() - st.session_state.last_refresh_time
    ).total_seconds()

    if (
        st.session_state.show_results
        or st.session_state.show_leaderboard
        or (auto_refresh and seconds_passed > refresh_interval)
    ):
        previous_results, _, _ = read_previous_results()

        if previous_results.shape[0] != 0:
            # Convert time to datetime
            previous_results["Time"] = pd.to_datetime(previous_results["Time"])

            # Merge previous results with the series DataFrame on specific columns
            # This combines the 'Series', 'p', and 'q' columns from both DataFrames
            # merged_df = pd.merge(previous_results.reset_index(), st.session_state.series,
            #                     on=previous_results.columns[1:-2].tolist(), how='left')# Sort both dataframes by the key columns
            previous_results = previous_results.reset_index().sort_values(
                by=previous_results.columns[1:-2].tolist()
            )
            series = st.session_state.series.sort_values(
                by=previous_results.columns[2:-2].tolist()
            )

            # Initialize a dictionary to store the results for each team and series
            team_results = {}

            # Group the data by Team and Series for each team's attempts on each series
            for team, team_data in previous_results.groupby("Team"):
                if team not in team_results:
                    team_results[team] = {}
                    
                team_data = team_data.sort_values("Time")
                for serie, attempts in team_data.groupby("Series"):
                    # Initialize counters
                    correct_attempts = 0
                    incorrect_attempts = 0
                    last_attempt_correct = False
                    weight_to_add = 0
                    
                    # Process each attempt
                    for _, attempt in attempts.iterrows():
                        # Find matching row in the series DataFrame
                        if config["arima_problem_type"] == "ARMA":
                            match = series[
                                (series["Series"] == attempt["Series"])
                                & (series["p"] == attempt["p"])
                                & (series["q"] == attempt["q"])
                                & (series["include_mean"] == attempt["include_mean"])
                            ]
                        elif config["arima_problem_type"] == "ARIMA":
                            match = series[
                                (series["Series"] == attempt["Series"])
                                & (series["p"] == attempt["p"])
                                & (series["d"] == attempt["d"])
                                & (series["q"] == attempt["q"])
                                & (series["include_mean"] == attempt["include_mean"])
                                & (np.isclose(series["lambda"], attempt["lambda"], atol=config["lambda_tol"]))
                            ]
                        else:
                            match = series[
                                (series['Series'] == attempt['Series']) &
                                (series['p'] == attempt['p']) &
                                (series['d'] == attempt['d']) &
                                (series['q'] == attempt['q']) &
                                (series['P'] == attempt['P']) &
                                (series['D'] == attempt['D']) &
                                (series['Q'] == attempt['Q']) &
                                (series['s'] == attempt['s']) &
                                (series['include_mean'] == attempt['include_mean']) &
                                (np.isclose(series['lambda'], attempt['lambda'], atol=config["lambda_tol"]))
                            ]
                        
                        if not match.empty:
                            correct_attempts += 1
                            last_attempt_correct = True
                            weight_to_add = match.iloc[0]['weight'] * 100
                        else:
                            incorrect_attempts += 1
                            last_attempt_correct = False
                    
                        # Store results for each series attempted by the team
                        team_results[team][attempt['Time']] = {
                            'Series': serie,
                            'correct': correct_attempts,
                            'incorrect': incorrect_attempts,
                            'last_correct': last_attempt_correct,
                            'n_tries': correct_attempts + incorrect_attempts,
                            'mark': weight_to_add if last_attempt_correct else 0
                        }

            # Convert the results dictionary to a DataFrame for easier analysis
            df = pd.DataFrame([
                {'Team': team, 'Time': time, **data}
                for team, time_data in team_results.items()
                for time, data in time_data.items()
            ])

            prev_series = "0"
            prev_mark = 0.0
            for idx, row in df.iterrows():
                if row["Series"] != prev_series:
                    prev_series = row["Series"]
                    prev_mark = 0.0
                if row["last_correct"] and prev_mark > 0:
                    df.at[idx, "mark"] = 0.0
                if only_last_try and not row["last_correct"] and prev_mark > 0:
                    df.at[idx, "mark"] = -prev_mark
                prev_mark = row["mark"]
                prev_series = row["Series"]

            df["cumsum_mark"] = df.groupby("Team").agg({"mark": "cumsum"})
            
            try:
                team_marks = df.drop_duplicates(subset=["Team", "Series"], keep="last")
                team_marks = team_marks.groupby("Team").agg({"cumsum_mark": "last", "correct": "sum", "incorrect": "sum", "n_tries": "sum", "Time": "last"}).reset_index()

                # Get the team with the highest mark that reached it the fastest
                winning_team = team_marks.sort_values(["cumsum_mark", "Time"], ascending=[False, True]).iloc[0]["Team"]

                df = df.sort_values(["Team", "Time"])
                df["mark"] = df.groupby("Team").agg({"mark": "cumsum"})

                
                # Pivot the DataFrame so that each 'Team' becomes a column
                barplot_df = df.set_index("Time").pivot(columns="Team", values="mark")

                if st.session_state.show_results:
                    n_steps = config["n_steps"]
                    # Create bar chart race animation
                    try:
                        html_str = bcr.bar_chart_race(
                            barplot_df.ffill().fillna(0),
                            interpolate_period=True,
                            steps_per_period=n_steps,
                            title="Team leaderboard evolution",
                            period_fmt="%H:%M:%S", 
                        ).data
                    except AttributeError:
                        html_str = bcr.bar_chart_race(
                            barplot_df.ffill().fillna(0),
                            interpolate_period=True,
                            steps_per_period=n_steps,
                            title="Team leaderboard evolution",
                            period_fmt="%H:%M:%S",
                        )

                    start = html_str.find("base64,") + len("base64,")
                    end = html_str.find('">')

                    video = base64.b64decode(html_str[start:end])

                    st.video(video)

                if st.session_state.show_leaderboard or auto_refresh:
                    # Sort the values in descending order
                    sorted_values = team_marks.sort_values(["cumsum_mark", "Time"], ascending=[True, True])[['Team', 'cumsum_mark']].set_index('Team')
                    sorted_values.columns = ["mark"]

                    # Get distinct colors for each bar using the viridis colormap
                    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_values)))

                    # Plotting the values
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # sorted_values.plot(y="mark",kind="barh", color=colors)
                    ax.barh(sorted_values.index, sorted_values["mark"], color=colors)
                    plt.title("Leaderboard")
                    plt.ylabel("Value")
                    plt.xlabel("Mark")
                    plt.xticks(rotation=0)
                    plt.grid(axis="x", linestyle="--", alpha=0.7)
                    plt.tight_layout()

                    st.pyplot(fig)

                # Show winning team name
                st.subheader(f"Winning team: {winning_team}")

                # Update last refresh time
                st.session_state.last_refresh_time = datetime.now()

                
                csv = convert_df(team_marks)

                # Get the series statistics
                series_stats = df.groupby("Series").agg({"correct": "sum", "incorrect": "sum", "n_tries": "sum"}).reset_index()
                series_csv = convert_df(series_stats)

                team_stats = df.groupby(["Team", "Series"]).agg({"correct": "sum", "incorrect": "sum", "n_tries": "sum", "last_correct": "last", "cumsum_mark": "last"}).reset_index()
                team_stats.columns = ["Team", "Series", "correct", "incorrect", "n_tries", "last_correct", "mark"]
                team_stats["mark"] = team_stats["mark"]
                team_stats_csv = convert_df(team_stats)

                st.download_button(
                    label="Download marks as CSV",
                    data=csv,
                    file_name="marks.csv",
                    mime="text/csv",
                )

                st.download_button(
                    label="Download Series Statistics",
                    data=series_csv,
                    file_name="series_stats.csv",
                    mime="text/csv",
                )
                
                st.download_button(
                    label="Download Teams Statistics",
                    data=team_stats_csv,
                    file_name="team_stats.csv",
                    mime="text/csv",
                )
            except IndexError:
                st.text("No points yet")
        else:
            st.text("No points yet")

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

    ## Read Admin account configuration
    admin_account = st.secrets["admin_account"]

    ## Read google sheet configuration
    google_sheets = st.secrets["google_sheets"]

    ## Read config
    config = st.secrets["config"]
    main()
