import os
import click
import time
from InquirerPy import inquirer



test = find_all()
for i in test:
    print(os.path.abspath(i))


print("\nWelcome to Learn2Slither project")
time.sleep(1)


try:
    choice_program = get_program()
    if choice_program == "Training":
        
except KeyboardInterrupt:
    click.echo(click.style("\nForce Quit...", fg='red'))
    return


def get_program():
    return inquirer.select(
        message="\n\nChoose Program",
        choices=["Training",
                 "Use"]
    ).execute()


def find_all():
    result = []
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for filename in files:
            if filename.endswith(".png"):
                result.append(os.path.basename(filename))
    return result
