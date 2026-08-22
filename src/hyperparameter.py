from InquirerPy import inquirer
import click


def create_hyperparameter():
    hyperparam = {
        'board_size': 10,
        'lr': 0.0001,
        'epsilon_decay': 0.99998,
        'batch_size': 128,
        'episode_train': 5001,
        'episode_apply': 501,
        'step_by_step': False,
        'visual': False,
        'Training': True
    }
    return hyperparam


def menuParameter(parameter: dict):
    return inquirer.select(
        message="\n\nparameter Information:\n"
        f"board_size : {parameter['board_size']}\n"
        f"Learning Rate : {parameter['lr']}\n"
        f"epsilon_decay : {parameter['epsilon_decay']}\n"
        f"batch_size : {parameter['batch_size']}\n"
        f"episode_train: {parameter['episode_train']}\n"
        f"episode_apply: {parameter['episode_apply']}\n"
        f"step_by_step: {parameter['step_by_step']}\n"
        f"visual: {parameter['visual']}\n\n"
        "Select Parameter",
        choices=["board_size", "Learning Rate", "epsilon_decay", "batch_size",
                 "episode_train", "episode_apply",
                 "activate_mask", "step_by_step", "visual", "Done"]
    ).execute()


def addParameter():
    parameter = create_hyperparameter()
    while (1):
        choice = menuParameter(parameter)
        if choice == "Done":
            return parameter
        elif choice == "step_by_step" or choice == "visual":
            parameter[choice] = new_param_bool()
        elif choice == "episode_train" or choice == "episode_apply" or\
                choice == "batch_size" or choice == "board_size":
            parameter[choice] = newParam_int(choice)
        else:
            parameter[choice] = newParam_float(choice)


def newParam_float(param):
    newChange = click.prompt(f"\nNew data for {param}", type=float)
    return newChange


def newParam_int(param):
    newChange = click.prompt(f"\nNew data for {param}", type=int)
    if param == 'board_size':
        while newChange <= 1:
            newChange = click.prompt(f"\nError, {param} must be > 1", type=int)
    return newChange


def new_param_bool():
    return inquirer.select(
        message="\n\n Activate {param}",
        choices=["True", "False"]
    ).execute()
