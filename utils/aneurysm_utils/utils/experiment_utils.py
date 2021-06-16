import os
import sys
import threading
import time

from aneurysm_utils.utils import text_utils


def in_ipython_environment():
    "Is the code running in the ipython environment (jupyter including)"

    program_name = os.path.basename(os.getenv("_", ""))

    if (
        "jupyter-notebook" in program_name
        or "ipython" in program_name  # jupyter-notebook
        or "JPY_PARENT_PID" in os.environ  # ipython
    ):  # ipython-notebook
        return True
    else:
        return False


def current_milli_time() -> int:
    return int(round(time.time() * 1000))


def call_function(func, **kwargs: dict):
    func_params = func.__code__.co_varnames[: func.__code__.co_argcount]
    func_args = {}
    for arg in kwargs:
        # arg or _arg in function parameters
        if arg in func_params or "_" + arg in func_params:
            func_args[arg] = kwargs[arg]
    return func(**func_args)


# Copied from: https://github.com/IDSIA/sacred/blob/master/sacred/utils.py#L678
class IntervalTimer(threading.Thread):
    @classmethod
    def create(cls, func, interval=10):
        stop_event = threading.Event()
        timer_thread = cls(stop_event, func, interval)
        return stop_event, timer_thread

    def __init__(self, event, func, interval=10.0):
        # TODO use super here.
        threading.Thread.__init__(self)
        self.stopped = event
        self.func = func
        self.interval = interval

    def run(self):
        while not self.stopped.wait(self.interval):
            self.func()
        self.func()


class StdoutFileRedirect:
    def __init__(self, log_path: str):
        # TODO check if sys already redirected sys.stdout.write.__name__
        self._stdout_current = sys.stdout.write
        self._stderr_current = sys.stderr.write
        self._log_path = log_path
        self._log_file = None

        def write_stdout(message):
            self._stdout_current(message)
            if message and len(message) > 1 and "varName" not in message:
                if not message.startswith("\n") and not message.startswith("\r"):
                    message = "\n" + message
                self.log_file.write(text_utils.safe_str(message))
                self.log_file.flush()

        self._write_stdout = write_stdout

        def write_stderr(message):
            self._stderr_current(message)
            if message and len(message) > 1 and "varName" not in message:
                if not message.startswith("\n") and not message.startswith("\r"):
                    message = "\n" + message
                self.log_file.write(text_utils.safe_str(message))
                self.log_file.flush()

        self._write_stderr = write_stderr

    @property
    def log_file(self):
        if self._log_file is None:
            if not os.path.exists(os.path.dirname(self._log_path)):
                os.makedirs(os.path.dirname(self._log_path))
            self._log_file = open(self._log_path, "a", encoding="utf-8")
        return self._log_file

    def redirect(self):
        sys.stdout.write = self._write_stdout
        sys.stderr.write = self._write_stderr

    def reset(self):
        # Reset writers
        sys.stdout.write = self._stdout_current
        sys.stderr.write = self._stderr_current

        # Close file and set None
        if self._log_file:
            try:
                self._log_file.close()
            except ValueError:
                pass
            self._log_file = None
