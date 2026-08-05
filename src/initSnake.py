import os


def find_all():
    result = []
    for root, dirs, files in os.walk("."):
        for filename in files:
            if filename.endswith(".py"):
                result.append(os.path.join(root))
    return result


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


print(find_all())
