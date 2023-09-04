def config_function():
    # Returns a dictionary containing the necessary information
    # for the tool
    output = {}
    
    ##############################################################
    ### Configuration to modify
    # Name of leaderboard file
    output['name_GDrive'] = 'Leaderboard_test' 
    
    # Define type of problem: "classification", "regression"
    output['type_problem'] = 'classification'
    
    # Define name of the Spreadsheet in GDrive that contains the team names and the pwd.
    output['pwd_table'] = 'Team_pwds'
    
    # Number of tries
    output['n_tries'] = 3
    ##############################################################
    
    # Select the columns of the leaderboard that will be shown:
    # - If type_problem == "classification", the valid value is "Accuracy"
    # - If type_problem == "regression", the valid values are "RMSE", "MAPE", and "Negative_Values"
    if output['type_problem'] == 'regression':
        output['col_show'] = ["RMSE", "MAPE", "Negative_Values"]
    else:
        output['col_show'] = ["Accuracy", 'Sensitivity', 'Specificity']
    # output['col_show'] = ["Accuracy"]

    # Define title of the webpage
    output['title'] = f"{output['type_problem'].capitalize()} Hackathon Machine Learning"

    
    # Define the name of the validation dataset to
    # be used.
    # - THE VALIDATION DATASET MUST BE ALWAYS STORED IN
    #   THE FOLDER 'Val_Data'
    # - The validation datasets should not contain any headers
    if output['type_problem'] == 'regression':
        name_fich = "DailyDemandTV.dat"
    else:
        name_fich = "ValidationY999_MIC.dat"
    output['val_dataset'] = f"Val_Data/{name_fich}"
    

    return output