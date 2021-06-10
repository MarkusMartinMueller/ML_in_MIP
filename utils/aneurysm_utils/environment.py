import datetime
import json
import logging
import os
import re
import sys
from enum import Enum, unique
from typing import Callable, List, Optional, Union

import comet_ml
import git
from addict import Dict

from aneurysm_utils.utils import (
    experiment_utils,
    file_utils,
    request_utils,
    text_utils,
)


@unique
class ExperimentState(str, Enum):
    INITIALIZED = "initialized"
    QUEUED = "queued"  # deprecated
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"

    def __str__(self):
        return str(self.value)


class Experiment:

    _STDOUT_FILE_NAME = "stdout.txt"
    _RUN_INFO_FILE_NAME = "{}_run.json"

    def __init__(
        self, env, name: str, comet_exp: comet_ml.Experiment, redirect_logs: bool = True
    ):
        # Initialize logger
        self.log = logging.getLogger(__name__)

        # Initialize internal variables -> Lazy initialization
        self._output_path: str = None
        self._running = False
        self._has_run = False

        # Initialize internal variables from parameters
        self._name: str = name
        self._env = env

        # Initialize components -> Lazy initialization
        self._stdout_file_redirect: Optional[experiment_utils.StdoutFileRedirect] = None

        # Initialize public variables
        self.init_time = datetime.datetime.now()
        self.comet_exp = comet_exp
        self.redirect_logs = redirect_logs
        self.params: dict = {}
        self.artifacts: dict = {}

        # Generate key
        timestamp = self.init_time.strftime("%Y-%m-%d-%H-%M-%S")
        self._key = "{}_{}".format(
            timestamp,
            # text_utils.simplify(self._env.operator),
            text_utils.simplify(self.name),
        )

        self.metadata = self._init_default_metadata()
        self.comet_exp.set_name(self._key)

        self.log.info("Experiment " + self.name + " is initialized.")
        self.log.debug("Experiment key: " + self.key)

    @property
    def name(self) -> str:
        """
        Returns the name of the experiment.
        """

        return self._name

    @property
    def key(self) -> str:
        """
        Returns the key of the experiment.
        """

        return self._key

    @property
    def output_path(self) -> str:
        """
        Returns the path to the root folder of the experiment.
        """
        if self._output_path is None:
            folder = os.path.join(self._env.experiments_folder, self.key)

            if not os.path.exists(folder):
                os.makedirs(folder)

            self._output_path = folder

        return self._output_path

    @property
    def stdout_path(self) -> str:
        """
        Returns the file path for the stdout logs.
        """
        return os.path.join(self.output_path, self._STDOUT_FILE_NAME)

    def run(
        self,
        exp_function: Union[List[Callable], Callable],
        params: dict = None,
        artifacts: dict = None,
    ):
        """
        Runs the given experiment function or list of functions and updates the experiment metadata.

        Args:
            exp_function: Method that implements the experiment.
            params: Dictionary that contains the configuration (e.g. hyperparameters) for the experiment.
            artifacts: Dictionary that contains artifacts (any kind of python object) required for the experiment.
        """
        # TODO track file events only during run?

        if self._running:
            self.log.warning(
                "Experiment is already running. Running same experiment in parallel "
                "might give some trouble."
            )
        elif self._has_run:
            self.log.info(
                "This experiment has already been run. Metadata will be overwritten! "
                "It is suggested to initialize a new experiment. "
                "The metadata of the last run is still saved in a run.json in the local exp folder."
            )

        # Redirect stdout/sterr to file
        if self._stdout_file_redirect is None:
            self._stdout_file_redirect = experiment_utils.StdoutFileRedirect(
                log_path=self.stdout_path
            )

        if self.redirect_logs:
            self._stdout_file_redirect.redirect()

        # Executing experiment functions
        self.log.info("Running experiment: " + self.key)
        self._running = True
        self._has_run = True  # even
        self._syncing = False

        # add artifacts
        if artifacts is not None:
            self.artifacts.update(artifacts)

        self.artifacts = artifacts

        # Log params
        if params is None:
            params = {}
        else:
            params = params.copy()

        self.params.update(params)
        self.comet_exp.log_parameters(self.params)

        # Wraps the experiment function into another function for more control
        def exp_wrapper():
            result_value = None
            kwargs = {
                "exp": self,
                "params": self.params,
                "artifacts": self.artifacts,
                "config": self.params,
                "log": self.log,
            }

            if type(exp_function) is list:
                for exp_func in exp_function:
                    if not callable(exp_func):
                        self.log.warning(str(exp_func) + " is not a function.")
                        continue
                    self.log.info("Running experiment function: " + exp_func.__name__)
                    if not self.metadata.command:
                        self.metadata.command = exp_func.__name__
                    else:
                        self.metadata.command += " -> " + exp_func.__name__

                    result_value = experiment_utils.call_function(exp_func, **kwargs)
            elif callable(exp_function):
                self.log.debug("Running experiment function: " + exp_function.__name__)
                self.metadata.command = exp_function.__name__
                result_value = experiment_utils.call_function(exp_function, **kwargs)
            else:
                self.log.error(str(exp_function) + " is not a function.")

            if result_value:
                return result_value

        # sync_heartbeat = None
        # heartbeat_stop_event = None

        # if self.auto_sync:
        # only required for auto sync
        # sync hearbeat for every 20 seconds - make configurable?
        # run sync exp without uploading resources
        #    (
        #        heartbeat_stop_event,
        #        sync_heartbeat,
        #    ) = experiment_utils.IntervalTimer.create(self.sync_exp, 20)
        # Initial sync of metadata
        #    self.sync_exp(upload_resources=False)

        try:
            # if sync_heartbeat:
            # Start heartbeat if initialized
            #    sync_heartbeat.start()
            self.metadata.status = ExperimentState.RUNNING
            self.set_completed(exp_wrapper())
        except:
            ex_type, val, tb = sys.exc_info()

            if ex_type is KeyboardInterrupt:
                # KeyboardInterrupt cannot be catched via except Exception: https://stackoverflow.com/questions/4990718/about-catching-any-exception
                self.metadata.status = ExperimentState.INTERRUPTED
            else:
                self.metadata.status = ExperimentState.FAILED

            self._finish_exp_run()

            # TODO: clean gpu memory
            raise
        finally:
            # always end comet experiment
            self.comet_exp.end()
            # if sync_heartbeat:
            # Always stop heartbeat if initialized
            #    heartbeat_stop_event.set()
            #    sync_heartbeat.join(timeout=2)

    def set_completed(self, result: Optional[str] = None):
        """
        Sets the experiment to completed and sync metadata and files if auto sync is enabled.
        Only required to manually complete experiment for example if run is not used.

        Args:
            result: Final result metric.
        """
        self._has_run = True
        self.metadata.result = result
        self.metadata.status = ExperimentState.COMPLETED

        self._finish_exp_run()

    # Internal Methods
    def _finish_exp_run(self):
        """
        Finishes experiment run and collects a few additional metadata.
        """
        self.metadata.finished_at = experiment_utils.current_milli_time()
        self.metadata.duration = int(
            round(self.metadata.finished_at - self.metadata.started_at)
        )

        state_desc = "finished"
        if self.metadata.status:
            state_desc = str(self.metadata.status)

        self.log.info(
            "Experiment run "
            + state_desc
            + ": "
            + self.name
            + "."
            + " Duration: "
            + text_utils.simplify_duration(self.metadata.duration)
        )

        self._running = False
        if self.redirect_logs and self._stdout_file_redirect:
            self._stdout_file_redirect.reset()

        # save experiment json for every run to local experiment folder
        run_json_name = self._RUN_INFO_FILE_NAME.format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )

        file_path = os.path.join(self.output_path, run_json_name)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "w+") as fp:
            json.dump(self.metadata.to_dict(), fp, sort_keys=True, indent=4)

        # sync experiment
        # if self.auto_sync:
        #    self.sync_exp(upload_resources=True)

    def _init_default_metadata(self) -> Dict:
        """
        Initializes the experiment metadata.
        """

        # TODO: use a more strictly defined data structure
        metadata = Dict()
        metadata.key = self._key
        metadata.name = self.name
        metadata.project = self._env.project
        metadata.started_at = experiment_utils.current_milli_time()
        metadata.status = ExperimentState.INITIALIZED
        return metadata


class RemoteFileHandler:
    _FILE_VERSION_SUFFIX_PATTERN = re.compile(r"\.v(\d+)$")

    def __init__(self, env):
        # Initialize logger
        self.log = logging.getLogger(__name__)

        # Initialize variables
        self.env = env

        # Files
        self._requested_files = []
        self._uploaded_files = []

        # File events
        self.on_file_requested = Event()
        self.on_file_uploaded = Event()

    @property
    def requested_files(self) -> list:
        """
        Returns the list of requested files (files requested via 'get_file()')
        """

        return self._requested_files

    @property
    def uploaded_files(self) -> list:
        """
        Returns the list of uploaded files (files uploaded via 'upload_file()')
        """
        return self._uploaded_files

    def get_file(
        self,
        key: str,
        force_download: bool = False,
        unpack: Optional[Union[bool, str]] = None,
        file_name: Optional[str] = None,
        track_event: bool = True,
    ) -> Optional[str]:
        """
        Returns local path to the file for the given `key`. If the file is not available locally, tries to download it from a connected remote storage.

        Args:
            key: Key or url of the requested file.
            force_download: If `True`, the file will always be downloaded and not loaded locally (optional).
            unpack: If `True`, the file - if a valid ZIP - will be unpacked within the data folder.
            If a path is provided, the ZIP compatible file will automatically be unpacked at the given path (optional).
            file_name: Filename to use locally to download the file (optional).
            track_event: If `True`, this file operation will be tracked and registered listeners will be notified (optional)/

        Returns:
            Local path to the requested file or `None` if file is not available.
        """

        file_updated = False
        resolved_key = key
        local_file_path: Optional[str]

        if request_utils.is_valid_url(key):
            if not request_utils.is_downloadable(key):
                self.log.warning(
                    "Key is a valid url, but cannot be downloaded. Trying to load local file."
                )
                if file_name:
                    local_file_path = self.load_local_file(
                        os.path.join(self.env.downloads_folder, file_name)
                    )
                    file_updated = False
            else:
                # key == url -> download from url
                local_file_path, file_updated = file_utils.download_file(
                    key,
                    self.env.downloads_folder,
                    file_name=file_name,
                    force_download=force_download,
                )
        elif os.path.isdir(self.resolve_path_from_key(key)):
            # If key resolves to a directory and not a file -> directly return directory
            return self.resolve_path_from_key(key)
        else:
            if file_name:
                local_file_path = self.load_local_file(file_name)
            else:
                local_file_path = self.load_local_file(key)

            if (
                not local_file_path
                or force_download
                or not os.path.isfile(local_file_path)
            ):
                # no local file found -> try to directly download file
                try:
                    self.log.warning(
                        "Failed to request file info from remote for key " + key
                    )
                    # TODO: local_file_path = self.download_file(key)

                    if local_file_path:
                        file_updated = True
                except Exception:
                    self.log.warning(
                        "Failed to request file info from remote for key " + key
                    )

        if not local_file_path:
            self.log.warning("Failed to find file for key: " + key)
            return None

        if not os.path.isfile(local_file_path):
            self.log.warning("File does not exist locally: " + local_file_path)
            return None

        if track_event:
            # use resolved key to get the dataset with version
            self._requested_files.append(resolved_key)
            self.on_file_requested(resolved_key)

        if unpack:
            remove_existing_folder = file_updated

            if isinstance(unpack, bool):
                unpack = os.path.join(
                    os.path.dirname(os.path.realpath(local_file_path)),
                    os.path.basename(local_file_path).split(".")[0],
                )

            # unpack is a path
            if not file_utils.is_subdir(str(unpack), self.env.root_folder):
                remove_existing_folder = False

            unpack_path = file_utils.unpack_archive(
                local_file_path, str(unpack), remove_existing_folder
            )
            if unpack_path and os.path.exists(unpack_path):
                return unpack_path
            else:
                self.log.warning(
                    "Unable to unpack file, its not a supported archive format: "
                    + local_file_path
                )
                return local_file_path

        return local_file_path

    def load_local_file(self, key: str) -> Optional[str]:
        """
        Loads a local file with the given key. Will also try to find the newest version if multiple versions exists locally for the key.

        Args:
            key (str): Key of the file.

        Returns:
            Path to file or `None` if no local file was found.
        """
        # always load latest version also from local
        local_file_path = self.resolve_path_from_key(key)

        file_dir = os.path.dirname(local_file_path)
        file_name = os.path.basename(local_file_path)

        if not os.path.isdir(file_dir):
            return None

        latest_file_path = None
        latest_file_version = 0

        for file in os.listdir(file_dir):
            if file.startswith(file_name):
                if self.get_version_from_key(file) > latest_file_version:
                    latest_file_path = os.path.abspath(os.path.join(file_dir, file))
                    latest_file_version = self.get_version_from_key(file)

        if latest_file_path and os.path.isfile(latest_file_path):
            self.log.debug(
                "Loading latest version ("
                + str(latest_file_version)
                + ") for "
                + key
                + " from local."
            )
            return latest_file_path

        return None

    def get_version_from_key(self, key: str) -> int:
        """
        Returns the version extracted from the key. Or 1 if no version is attached.
        """
        version_suffix = self._FILE_VERSION_SUFFIX_PATTERN.search(key)
        if version_suffix:
            return int(version_suffix.group(1))
        else:
            return 1

    def resolve_path_from_key(self, key: str) -> str:
        """
        Returns the local path for a given key.
        """
        return os.path.abspath(os.path.join(self.env.project_folder, key))


# https://stackoverflow.com/questions/1092531/event-system-in-python
# Use library instead? https://zopeevent.readthedocs.io/en/latest/index.html


class Event(list):
    """Event subscription.

    A list of callable objects. Calling an instance of this will cause a
    call to each item in the list in ascending order by index.

    Example Usage:
    >>> def f(x):
    ...     print 'f(%s)' % x
    >>> def g(x):
    ...     print 'g(%s)' % x
    >>> e = Event()
    >>> e()
    >>> e.append(f)
    >>> e(123)
    f(123)
    >>> e.remove(f)
    >>> e()
    >>> e += (f, g)
    >>> e(10)
    f(10)
    g(10)
    >>> del e[0]
    >>> e(2)
    g(2)

    """

    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)


class Environment:
    _ENV_NAME_ENV_ROOT_PATH = "DATA_ENVIRONMENT"
    _TEMP_ROOT_FOLDER = "temp"

    # local folders
    _LOCAL_ENV_FOLDER_NAME = "environment"
    _EXPERIMENTS_FOLDER_NAME = "experiments"
    _DATASETS_FOLDER_NAME = "datasets"
    _MODELS_FOLDER_NAME = "models"
    _DOWNLOADS_FOLDER_NAME = "downloads"

    _LOCAL_OPERATOR = "local"
    _LOCAL_PROJECT = "local"

    def __init__(self, project: str, root_folder: str = None):

        # Create the Logger
        self.log = logging.getLogger(__name__)

        # Initialize parameters
        self._file_handler: Optional[RemoteFileHandler] = None
        self._project: str = project
        self._cached_data: dict = {}

        # Get repo root path
        try:
            self._repo_folder = git.Repo(
                "", search_parent_directories=True
            ).working_tree_dir
        except Exception:
            # not a git reposiotry, use current working directory
            self._repo_folder = os.getcwd()

        # Set root folder
        if not root_folder:
            # use environment variable
            root_folder = os.getenv(self._ENV_NAME_ENV_ROOT_PATH)

        if not root_folder:
            # that current git root as environment folder
            root_folder = os.path.join(self.repo_folder, self._LOCAL_ENV_FOLDER_NAME)

        if not root_folder:
            # create local environment
            root_folder = self._LOCAL_ENV_FOLDER_NAME

        if root_folder == self._TEMP_ROOT_FOLDER:
            # if folder is temp -> create temporary folder that will be removed on exit
            import tempfile
            import atexit
            import shutil

            root_folder = tempfile.mkdtemp()

            # automatically remove temp directory if process exits
            def cleanup():
                self.log.info("Removing temp directory: " + root_folder)
                shutil.rmtree(root_folder)

            atexit.register(cleanup)

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        self._root_folder = root_folder

    def print_info(self):
        """
        Prints out a summary of the configuration of the environment instance. Can be used as watermark for notebooks.
        """
        print("Environment Info:")
        print("")
        from aneurysm_utils.__version__ import __version__

        print("Library Version: " + str(__version__))
        print("Configured Project: " + self.project)
        # print("Configured Operator: " + self.operator)
        print("")
        print("Folder Structure: ")
        print("- Root folder: " + os.path.abspath(self.root_folder))
        print(" - Project folder: " + self.project_folder)
        print(" - Datasets folder: " + self.datasets_folder)
        print(" - Models folder: " + self.models_folder)
        print(" - Experiments folder: " + self.experiments_folder)

    @property
    def project(self) -> str:
        """
        Returns the name of the configured project.
        """

        if self._project is None:
            self._project = self._LOCAL_PROJECT

        return self._project

    @property
    def repo_folder(self) -> str:
        """
        Returns the path to the project repository if it is run within a git repo.
        """

        return self._repo_folder

    @property
    def root_folder(self) -> str:
        """
        Returns the path to the root folder of the environment.
        """

        return self._root_folder

    @property
    def project_folder(self) -> str:
        """
        Returns the path to the project folder of the environment.
        """
        folder = os.path.join(self.root_folder, self.project) # , text_utils.simplify(self.project))

        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

    @property
    def datasets_folder(self) -> str:
        """
        Returns the path to the datasets folder of the selected project.
        """
        folder = os.path.join(self.project_folder, self._DATASETS_FOLDER_NAME)

        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

    @property
    def models_folder(self) -> str:
        """
        Returns the path to the models folder of the selected project.
        """
        folder = os.path.join(self.project_folder, self._MODELS_FOLDER_NAME)

        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

    @property
    def downloads_folder(self) -> str:
        """
        Returns the path to the downloads folder of the selected project. This folder contains downloaded via url.
        """
        folder = os.path.join(self.project_folder, self._DOWNLOADS_FOLDER_NAME)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    @property
    def experiments_folder(self) -> str:
        """
        Returns the path to the experiment folder of the environment.
        """
        folder = os.path.join(self.project_folder, self._EXPERIMENTS_FOLDER_NAME)
        if not os.path.exists(folder):
            os.makedirs(folder)

        return folder

        # Handlers

    @property
    def file_handler(self) -> RemoteFileHandler:
        """
        Returns the file handler. The file handler provides additional functionality for interacting with the remote storage.
        """

        if self._file_handler is None:
            self._file_handler = RemoteFileHandler(self)

        return self._file_handler

    @property
    def cached_data(self) -> dict:
        return self._cached_data

    def get_file(
        self,
        key: str,
        force_download: bool = False,
        unpack: Optional[Union[bool, str]] = None,
        file_name: Optional[str] = None,
        track_event: bool = True,
    ) -> Optional[str]:
        """
        Returns local path to the file for the given `key`. If the file is not available locally, tries to download it from a connected remote storage.

        Args:
            key: Key or url of the requested file.
            force_download: If `True`, the file will always be downloaded and not loaded locally (optional).
            unpack: If `True`, the file - if a valid ZIP - will be unpacked within the data folder.
            If a path is provided, the ZIP compatible file will automatically be unpacked at the given path (optional).
            file_name: Filename to use locally to download the file (optional).
            track_event: If `True`, this file operation will be tracked and registered listeners will be notified (optional)/

        Returns:
            Local path to the requested file or `None` if file is not available.
        """

        return self.file_handler.get_file(
            key,
            force_download=force_download,
            unpack=unpack,
            file_name=file_name,
            track_event=track_event,
        )

    def create_experiment(
        self, name: str, comet_exp: comet_ml.Experiment, **kwargs
    ) -> Experiment:
        """
        Creates a new experiment and saves it as active experiment.

        Args:
            name: Short description of the experiment.
            comet_exp: Comet.ml experiment.

        Returns:
            The created Experiment instance.
        """
        self.active_exp = Experiment(self, name, comet_exp=comet_exp, **kwargs)
        return self.active_exp
