
import storage

def compute_regressions(bao_reg):
    total_regressed = 0
    total_regression = 0
    for plan_group in storage.experiment_results():
        plan_group = list(plan_group)
        plans = [x["plan"] for x in plan_group]
        best_latency = min(plan_group, key=lambda x: x["reward"])["reward"]
        
        if bao_reg:
            selection = bao_reg.predict(plans).argmin()
        else:
            # If bao_reg is false-y, compare against PostgreSQL.
            selection = 0
                
        selected_plan_latency = plan_group[selection]["reward"]
        
        # Check to see if the regression is more than 1%.
        if selected_plan_latency > best_latency * 1.01:
            total_regressed += 1

        total_regression += selected_plan_latency - best_latency

    return (total_regressed, total_regression)


def should_replace_model(old_model, new_model):
    # Check the trained model for regressions on experimental queries.
    new_num_reg, new_reg_amnt = compute_regressions(new_model)
    cur_num_reg, cur_reg_amnt = compute_regressions(old_model)

    print("Old model # regressions:", cur_num_reg,
          "regression amount:", cur_reg_amnt)
    print("New model # regressions:", new_num_reg,
          "regression amount:", new_reg_amnt)

    # If our new model has no regressions, always accept it.
    # Otherwise, see if our regression profile is strictly better than
    # the previous model.
    if new_num_reg == 0:
        print("New model had no regressions.")
        return True
    elif cur_num_reg >= new_num_reg and cur_reg_amnt >= new_reg_amnt:
        print("New model with better regression profile",
              "than the old model.")
        return True
    else:
        print("New model did not have a better regression profile.")
        return False