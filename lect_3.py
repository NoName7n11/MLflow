import mlflow
import pandas as pd

mlflow.set_experiment("Lect_3")

with mlflow.start_run(run_name="Run_1", run_id="ce9e0f359cdd4bde9c411b49861e2692"):
    # There are three core components used to configure, track, and evaluate processes:
        # 1) Parameters
        # 2) Metrics
        # 3) Artifacts


    #Parameters: There are two ways to log parameters:
        #1) Key-value parameters
        #2) Dictionary parameters
    
    #Key-value parameters
    mlflow.log_param("learning _rate", 0.01)
    mlflow.log_param("epoch", 100)

    #Dictionary parameters
    parameters = {
        "learning_rate_1" : 0.04,
        "epoch_1" : 200
    }
    #Now the above dictionary parameters can be logged using the following command:
    mlflow.log_params(parameters)


    #Metrics: There are two ways to log metrics:
        #1) Key-value metrics
        #2) Dictionary metrics

    #Key-value metrics
    mlflow.log_metric("accuracy", 0.80)

    #Dictionary parameters
    metrics = {
        "accuracy_1" : 0.85,
        "accuracy_2" : 0.90
    }
    mlflow.log_metrics(metrics)


    #Artifacts: There are two ways to log artifacts:
        #1) Log a single artifact
        #2) Log multiple artifacts

    #Log a single artifact
    artifact_path = "img.png"
    mlflow.log_artifact(artifact_path)

    #there some other artifacts that can be logged such as:
        #1) mlflow.log_image()
        #2) mlflow.log_table()

    #mlflow.log_table()
    demo_df = pd.DataFrame({
        "name" : ["Naruto", "Ichigo", "Midoriya", "Aizen", "Tanjiro"],
        "power" : ["Rasengan", "Getsuga Tensho", "States of Smash", "Kyōka Suigetsu", "Hiokami Kagura"]
    })
    mlflow.log_table(demo_df, "demo_of_df.json")

    pokemon_dataset = pd.read_csv("C:\\Users\\novan\\Desktop\\ALL\\To_Learn\\MLFlow\\Dataset\\EN_Card_Data.csv")
    mlflow.log_table(pokemon_dataset, "Pokemon.json")