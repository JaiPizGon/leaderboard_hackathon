def config_function():
    # Returns a dictionary containing the necessary information
    # for the tool
    output = {}
    
    ##############################################################
    ### Configuration to modify
    # Name of leaderboard file
    output['name_GDrive'] = 'ARIMA_test' 

    
    # Define name of the Spreadsheet in GDrive that contains the team names and the pwd.
    output['pwd_table'] = 'Team_pwds'
    
    # Define which type of problem are you dealing with: ARMA, ARIMA, SARIMA
    output['problem_type'] = 'ARMA'
    
    # Define the number of tries the team has for each ARMA series
    output['n_tries'] = 1
    
    # Define the number of steps between frames in champion animation
    output['n_steps'] = 10
    
    ##############################################################
    # Name of ARIMA series file
    output['name_Arima'] = f'{output["name_GDrive"]}_Series'

    return output