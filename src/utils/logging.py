import sys
from colorama import Fore, Style


def log_error(text: str):
    sys.stderr.write(Fore.RED + Style.BRIGHT + text + Style.RESET_ALL + "\n")
