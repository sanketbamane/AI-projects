import sys

# Ensure we're running on Python 3.12.x where pyaudio (microphone) works.



def print_hi(name):
    print(f'Hi, {name}')


# Press the green button in the gutter to run the script.
def main():
    """Launch the voice interview CLI."""
    try:
        from voicebot.cli import main as cli_main
    except Exception:
        print("voicebot package not found or failed to import. Make sure the package files are present.")
        return
    cli_main()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
