import os


async def read_accuracy(folder_name):
    script_dir = os.getcwd()
    path = os.path.join(script_dir,"uploads")
    path = os.path.join(path,folder_name)
    path = os.path.join(path, "accuracy.txt")
    print(path)
    if os.path.exists(path):
        with open(path, "r") as file:
            accuracy = float(file.read().strip())
        return accuracy
    else:
            # print(f"Waiting for {path}...")
        return {"wait"}


