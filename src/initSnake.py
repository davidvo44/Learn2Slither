import os
import click
import time
from InquirerPy import inquirer
from src.learnSnake import learnSnake
from src.apply_nn import use
from pathlib import Path
from src.hyperparameter import addParameter
import torch


def main():
    print("\nWelcome to Learn2Slither project")

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA disponible  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Nom du GPU      : {torch.cuda.get_device_name(0)}")
    time.sleep(1)

    try:
        choice_program = get_program()
        if choice_program == "Training":
            train_prog = get_training_format()
            if train_prog == "From Scratch":
                create_loop()
            else:
                filepath = get_file()
                param = addParameter()
                learnSnake(param, filepath)
        else:
            filepath = get_file()
            param = addParameter()
            use(param, filepath)

    except KeyboardInterrupt:
        click.echo(click.style("\nForce Quit...", fg='red'))
        return


def get_program():
    return inquirer.select(
        message="\n\nChoose Program",
        choices=["Training",
                 "Use"]
    ).execute()


def get_training_format():
    return inquirer.select(
        message="\n\nFormat Training?",
        choices=["From Scratch",
                 "Use Checkpoint"]
    ).execute()


def get_file():
    all_file = find_all()
    if all_file:
        choice = select_file(all_file)
        return choice
    else:
        print('No file Found')
        return None


def find_all():
    result = []
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for filename in files:
            if filename.endswith(".pth"):
                result.append(os.path.join(root, filename))
    return result


def select_file(all_file):
    return inquirer.select(
        message='Select File\n',
        choices=all_file
        ).execute()


def create_loop():
    filescore = "None"
    param = addParameter()
    board = param["board_size"]
    for i in range(1000):
        param["board_size"] = board
        if filescore != "None":
            learnSnake(param, filescore)
        else:
            learnSnake(param)
        filenames = []
        best_score = float('-inf')
        base_dir = Path(__file__).parent
        filenames = [
            str(path) for path in base_dir.rglob("*.pth")
            if "learning_data" in path.parts
        ]
        filescore = "None"
        for file in filenames:
            score = use(param, file, learn=True)
            print(file, '     score:%.2f' % score)
            if score > best_score:
                if filescore and os.path.exists(filescore):
                    os.remove(filescore)
                best_score = score
                filescore = file
            else:
                if os.path.exists(file):
                    os.remove(file)

        print('SAVE:', filescore)
        file_path = Path(filescore)  # Ex: "src/learning_data/model.pth"

        if file_path.exists():
            new_path = file_path.with_stem(file_path.stem + "_save")
            file_path.replace(new_path)
        filescore = str(new_path)
        board += 1
        if board == 50:
            board = 3


if __name__ == "__main__":
    main()
