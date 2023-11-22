import pandas as pd
import torch

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from models.helper.training import step_train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURR_MODEL = "resnet18"


def objective_function(params):
    return step_train_model(params, pt_model=CURR_MODEL)


if __name__ == "__main__":
    space = [
        Categorical([16, 32, 64, 128], name='batch_size'),
        Integer(3, 6, name='freez_ep'),
        Integer(5, 10, name='unfreez_ep'),
        Real(0.0001, 0.001, "log-uniform", name='lr_1'),
        Real(0.00001, 0.0005, "log-uniform", name='lr_2'),
        Real(0.2, 0.8, name='dropout_head'),
        Categorical([512, 256, 128, 64], name='hidden_layer_1'),
        Categorical([512, 256, 128, 64], name='hidden_layer_2'),
        Categorical([0, 1, 2], name='num_hidden_layers'),
    ]

    res = gp_minimize(objective_function, space, n_calls=20, random_state=64)

    df = pd.DataFrame(res.x_iters,
                      columns=["batch_size", "freez_ep", "unfreez_ep", "lr_1", "lr_2", "dropout_head", "hidden_layer_1",
                               "hidden_layer_2", "num_hidden_layers"])
    df['function_values'] = res.func_vals
    csv_file_path = f'./{CURR_MODEL}/cache/hpt_resnet18_results.csv'
    df.to_csv(csv_file_path, index=False)

    score, model = step_train_model(
        res.x,
        pt_model=CURR_MODEL,
        return_model=True)
    torch.save(model.state_dict(), f'./{CURR_MODEL}/cache/resnet18.pth')

    print(f"Best parameters: {res.x}")
    print(f"Best score: {res.fun}")
