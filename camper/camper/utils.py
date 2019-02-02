from datetime import datetime


def folder_date():
    return datetime.now().strftime('%m-%d_%H-%M-%S')
