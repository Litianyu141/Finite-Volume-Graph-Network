import os
import time
import torch
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
from natsort import natsorted
import pickle
import json
import shutil


class Logger:

    def __init__(
        self,
        name,
        datetime=None,
        use_csv=False,
        use_tensorboard=True,
        params=None,
        git_info=None,
        saving_path=None,
        copy_code=True,
    ):
        """
        Logger logs metrics to CSV files / tensorboard
        :name: logging name (e.g. model name / dataset name / ...)
        :datetime: date and time of logging start (useful in case of multiple runs). Default: current date and time is picked
        :use_csv: log output to csv files (needed for plotting)
        :use_tensorboard: log output to tensorboard
        """
        self.name = name
        self.params = params
        if datetime:
            self.datetime = datetime
        else:
            self.datetime = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if saving_path is not None:
            self.saving_path = saving_path
        else:
            self.saving_path = os.getcwd() + f"/Logger/{name}/{self.datetime}"

        source_valid_file_path = os.path.split(os.path.split(__file__)[0])[0]
        target_valid_file_path = self.saving_path
        if copy_code:
            shutil.copytree(
                source_valid_file_path,
                target_valid_file_path,
                ignore=self.ignore_files_and_folders,
            )
        # shutil.copyfile(source_valid_file_path,target_valid_file_path)
        self.target_valid_file_path = target_valid_file_path + "/validate.py"
        self.use_tensorboard = use_tensorboard

        self.use_csv = use_csv
        if use_csv:
            os.makedirs("Logger/{}/{}/logs".format(name, self.datetime), exist_ok=True)
            os.makedirs("Logger/{}/{}/plots".format(name, self.datetime), exist_ok=True)

        if use_tensorboard:
            directory = self.saving_path + "/tensorboard"
            os.makedirs(directory, exist_ok=True)
            self.writer = SummaryWriter(directory)
        self.git_info = git_info

    def ignore_files_and_folders(self, dir_name, names):
        ignored = set()
        files_to_ignore = {}
        folders_to_ignore = {"rollout"}

        for name in names:
            path = os.path.join(dir_name, name)

            if os.path.isfile(path) and name in files_to_ignore:
                ignored.add(name)
            elif os.path.isdir(path) and name in folders_to_ignore:
                ignored.add(name)

        return ignored

    def log(self, item, value, index):
        """
        log index value couple for specific item into csv file / tensorboard
        :item: string describing item (e.g. "training_loss","test_loss")
        :value: value to log
        :index: index (e.g. batchindex / epoch)
        """

        if self.use_csv:
            filename = "Logger/{}/{}/logs/{}.log".format(self.name, self.datetime, item)

            if os.path.exists(filename):
                append_write = "a"
            else:
                append_write = "w"

            with open(filename, append_write) as log_file:
                log_file.write("{}, {}\n".format(index, value))

        if self.use_tensorboard:
            self.writer.add_scalar(item, value, index)

    def log_histogram(self, item, values, index):
        """
        log index values-histogram couple for specific item to tensorboard
        :item: string describing item (e.g. "training_loss","test_loss")
        :values: values to log
        :index: index (e.g. batchindex / epoch)
        """
        if self.use_tensorboard:
            self.writer.add_histogram(item, values, index)

    def log_model_gradients(self, item, model, index):
        """
        log index model-gradients-histogram couple for specific item to tensorboard
        :item: string describing model item (e.g. "encoder","discriminator")
        :values: values to log
        :index: index (e.g. batchindex / epoch)
        """
        if self.use_tensorboard:
            params = [p for p in model.parameters()]
            if len(params) != 0:
                gradients = torch.cat(
                    [p.grad.view(-1) for p in params if p.grad is not None]
                )
                self.writer.add_histogram(f"{item}_grad_histogram", gradients, index)
                self.writer.add_scalar(f"{item}_grad_norm2", gradients.norm(2), index)

    def plot(self, item, log=False, smoothing=0.025, ylim=None):
        """
        plot item metrics
        :item: item
        :log: logarithmic scale. Default: False
        :smoothing: smoothing of metric. Default: 0.025
        :ylim: y-axis limits [lower,upper]
        """
        if self.use_csv:
            plt.figure(1, figsize=(12, 6))
            plt.clf()
            plt.title(self.name)
            plt.ylabel(item)
            plt.xlabel("index")
            train_loss = np.loadtxt(
                open(
                    "Logger/{}/{}/logs/{}.log".format(self.name, self.datetime, item),
                    "rb",
                ),
                delimiter=",",
            )
            if log:
                plt.semilogy(train_loss[:, 0], train_loss[:, 1], "r")
            else:
                plt.plot(train_loss[:, 0], train_loss[:, 1], "r")
            train_loss = lowess(
                train_loss[:, 1], train_loss[:, 0], is_sorted=True, frac=smoothing, it=0
            )
            if log:
                plt.semilogy(train_loss[:, 0], train_loss[:, 1], "b")
            else:
                plt.plot(train_loss[:, 0], train_loss[:, 1], "b")
            mean = np.mean(train_loss[:, 1])
            std = np.std(train_loss[:, 1])
            if log:
                plt.savefig(
                    "Logger/{}/{}/plots/{}_log.png".format(
                        self.name, self.datetime, item
                    ),
                    dpi=400,
                )
            else:
                if ylim is not None:
                    plt.ylim(ylim)
                else:
                    try:
                        plt.ylim([mean - 2 * std, mean + 4 * std])
                    except:
                        pass
                plt.savefig(
                    "Logger/{}/{}/plots/{}.png".format(self.name, self.datetime, item),
                    dpi=400,
                )

        else:
            warnings.warn("set use_csv=True if you want to plot metrics")

    def save_state(
        self,
        model,
        optimizer,
        scheduler,
        index="final",
        trian_time_steps: np.ndarray = None,
    ):
        """
        saves state of model and optimizer
        :model: model to save (if list: save multiple models)
        :optimizer: optimizer (if list: save multiple optimizers)
        :index: index of state to save (e.g. specific epoch)
        """
        os.makedirs(self.saving_path + "/states", exist_ok=True)
        path = self.saving_path + "/states"

        with open(path + "/commandline_args.json", "wt") as f:
            json.dump(
                {**vars(self.params), **self.git_info}, f, indent=4, ensure_ascii=False
            )

        model.save_checkpoint(
            path + "/{}.state".format(index), optimizer, scheduler, trian_time_steps
        )
        return path + "/{}.state".format(index)

    def save_dict(self, dic, index="final"):
        """
        saves dictionary - helpful to save the population state of an evolutionary optimization algorithm
        :dic: dictionary to store
        :index: index of state to save (e.g. specific evolution)
        """
        os.makedirs(
            "Logger/{}/{}/states".format(self.name, self.datetime), exist_ok=True
        )
        path = "Logger/{}/{}/states/{}.dic".format(self.name, self.datetime, index)
        with open(path, "wb") as f:
            pickle.dump(dic, f)

    def load_state(
        self,
        model,
        optimizer,
        scheduler,
        datetime=None,
        index=None,
        continue_datetime=False,
        device=None,
        trian_time_steps=None,
    ):
        """
        loads state of model and optimizer
        :model: model to load (if list: load multiple models)
        :optimizer: optimizer to load (if list: load multiple optimizers; if None: don't load)
        :datetime: date and time from run to load (if None: take latest folder)
        :index: index of state to load (e.g. specific epoch) (if None: take latest index)
        :continue_datetime: flag whether to continue on this run. Default: False
        :return: datetime, index (helpful, if datetime / index wasn't given)
        """

        if datetime is None:
            for _, dirs, _ in os.walk("Logger/{}/".format(self.name)):
                datetime = sorted(dirs)[-1]
                if datetime == self.datetime:
                    datetime = sorted(dirs)[-2]
                break

        if continue_datetime:
            # CODO: remove generated directories...
            os.rmdir()
            self.datetime = datetime

        if index is None:
            for _, _, files in os.walk(
                "Logger/{}/{}/states/".format(self.name, datetime)
            ):
                index = os.path.splitext(natsorted(files)[-1])[0]
                break

        path = "Logger/{}/{}/states/{}.state".format(self.name, datetime, index)

        trian_time_steps = model.load_checkpoint(
            optimizer=optimizer,
            scheduler=scheduler,
            ckpdir=path,
            device=device,
            trian_time_steps=trian_time_steps,
        )

        return datetime, index, trian_time_steps

    def load_dict(self, dic, datetime=None, index=None, continue_datetime=False):
        """
        loads state of model and optimizer
        :dic: (empty) dictionary to fill with state information
        :datetime: date and time from run to load (if None: take latest folder)
        :index: index of state to load (e.g. specific epoch) (if None: take latest index)
        :continue_datetime: flag whether to continue on this run. Default: False
        :return: datetime, index (helpful, if datetime / index wasn't given)
        """

        if datetime is None:
            for _, dirs, _ in os.walk("Logger/{}/".format(self.name)):
                datetime = sorted(dirs)[-1]
                if datetime == self.datetime:
                    datetime = sorted(dirs)[-2]
                break

        if continue_datetime:
            # CODO: remove generated directories...
            os.rmdir()
            self.datetime = datetime

        if index is None:
            for _, _, files in os.walk(
                "Logger/{}/{}/states/".format(self.name, datetime)
            ):
                index = os.path.splitext(natsorted(files)[-1])[0]
                break

        path = "Logger/{}/{}/states/{}.dic".format(self.name, datetime, index)
        with open(path, "rb") as f:
            state = pickle.load(f)

        for key in state.keys():
            dic[key] = state[key]

        return datetime, index

    def load_logger(self, datetime=None, load=False):
        """
        copy older tensorboard logger to new dir
        :datetime: date and time from run to load (if None: take latest folder)
        """

        if datetime is None:
            for _, dirs, _ in os.walk("Logger/{}/".format(self.name)):
                datetime = sorted(dirs)[-1]
                if datetime == self.datetime:
                    datetime = sorted(dirs)[-2]
                break

        if load:

            cwd = os.getcwd()
            path = "Logger/{0}/{1}/tensorboard/".format(self.name, datetime)
            for _, _, files in os.walk(path):
                for file in files:
                    older_tensorboard_n = file
                    older_tensorboard = path + older_tensorboard_n

                    newer_tensorboard = (
                        cwd
                        + "/Logger/{0}/{1}/tensorboard/".format(
                            self.name, self.datetime
                        )
                        + older_tensorboard_n
                    )
                    shutil.copyfile(older_tensorboard, newer_tensorboard)
                break

            if os.path.exists(newer_tensorboard):
                print(
                    "older tensorboard aleady been copied to {0}".format(
                        newer_tensorboard
                    )
                )


t_start = 0


def t_step():
    """
    returns delta t from last call of t_step()
    """
    global t_start
    t_end = time.perf_counter()
    delta_t = t_end - t_start
    t_start = t_end
    return delta_t
