#In this we will learn about "Nested Runs" in MLflow

import mlflow 

mlflow.set_experiment("Nested_Run_Flow")

with mlflow.start_run(run_name = "Parent_Run") as parent_run:
    print(f"Parent run: {parent_run.info.run_id}")
    mlflow.log_param("theta", 100)  #Parent_1

    with mlflow.start_run(run_name = "Child_Run_1", nested = True) as child_run_1:
        mlflow.log_param()
        print(f"Child_Run_1: {child_run_1.info.run_id}")  # Child_1

    with mlflow.start_run(run_name = "Child_Run_2", nested = True) as child_run_2:
        print(f"Child_Run_2: {child_run_2.info.run_id}")  # Child_2

    with mlflow.start_run(run_name = "Child_Run_3", nested = True) as child_run_3:
        print(f"Child_Run_3: {child_run_3.info.run_id}")  #Child_3

# Nested run is used to group related experiments togather raher than cluttering the MLflow UI
# 


