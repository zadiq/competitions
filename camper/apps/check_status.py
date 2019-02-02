from camper.checkpoints import StatusLogCheckpoint

if __name__ == '__main__':
    path = input('Provide path to log. >> ')
    while True:
        _ = input('Press Enter for updates. >>')
        logger = StatusLogCheckpoint(path, read_only=True)
        logger.display_all()

