from InquirerPy import inquirer
import click


def create_hyperparameter():
    hyperparam = {
        'lr': 0.0005,
        'epsilon_decay': 0.99998,
        'batch_size': 128,
        'episode_train': 5001,
        'episode_apply': 501,
        'step_by_step': False,
        'visual': False,
        'Training': True,
        'board_size': 10
    }
    return hyperparam


def menuParameter(parameter: dict):
    return inquirer.select(
        message="\n\nparameter Information:\n"
        f"Learning Rate : {parameter['lr']}\n"
        f"epsilon_decay : {parameter['epsilon_decay']}\n"
        f"batch_size : {parameter['batch_size']}\n"
        f"episode_train: {parameter['episode_train']}\n"
        f"episode_apply: {parameter['episode_apply']}\n"
        f"step_by_step: {parameter['step_by_step']}\n"
        f"board_size: {parameter['board_size']}\n"
        f"visual: {parameter['visual']}\n\n"
        "Select Parameter",
        choices=["Learning Rate", "epsilon_decay", "batch_size",
                 "episode_train", "episode_apply",
                 "activate_mask", "step_by_step",
                 "visual", "board_size", "Done"]
    ).execute()


def addParameter():
    parameter = create_hyperparameter()
    while (1):
        choice = menuParameter(parameter)
        if choice == "Done":
            return parameter
        elif choice == "step_by_step" or choice == "visual":
            parameter[choice] = new_param_bool()
        else:
            parameter[choice] = newParam(choice)


def newParam(param):
    if param == "Learning Rate" or param == "epsilon_decay":
        newChange = click.prompt(f"\nNew data for {param}", type=float)
    else:
        newChange = click.prompt(f"\nNew data for {param}", type=int)
    return newChange


def new_param_bool():
    return inquirer.select(
        message="\n\n Activate {param}",
        choices=["True", "False"]
    ).execute()
