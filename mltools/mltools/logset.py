# -*- coding:utf-8 -*-

import numpy as np
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob


class LogSet:
    def __init__(self, data_source, settings=None):
        if isinstance(data_source, dict):
            self.data = data_source

        elif os.path.isdir(data_source):
            filelist = glob(
                os.path.join(data_source, "**", "*_log.json"), recursive=True
            )
            if settings is None:
                raise ValueError("settings must be specified.")

            self.data = {}
            self.data["summary"] = False
            self.data["xaxis"] = None
            self.data["values"] = {}
            for f in filelist:
                for t in settings["data-types"]:
                    typename = t["typename"]
                    if not t["filename_includes"] in f:
                        continue
                    if not typename in self.data["values"]:
                        self.data["values"][typename] = {}
                    log_file = json.load(open(f, "r"))
                    for column in log_file["log"]:
                        log = np.array(log_file["log"][column])
                        if self.data["xaxis"] is None:
                            self.data["xaxis"] = log[:, 0].T.tolist()
                        data_table = np.squeeze(log[:, 1:]).tolist()
                        if not column in self.data["values"][typename]:
                            self.data["values"][typename][column] = [data_table]
                        else:
                            self.data["values"][typename][column].append(data_table)

        elif os.path.isfile(data_source) and ".json" in data_source:
            self.data = json.load(open(data_source, "r"))

        else:
            raise ValueError("unknown data source.")

    def summary(self, npfunc=np.average):
        if self.data["summary"]:
            raise ValueError("data is already summarized.")
        summary_data = {}
        summary_data["summary"] = True
        summary_data["xaxis"] = self.data["xaxis"]
        summary_data["values"] = {}
        for t in self.data["values"]:
            summary_data["values"][t] = {}
            for c in self.data["values"][t]:
                datas = np.array(self.data["values"][t][c])
                summary_data["values"][t][c] = npfunc(datas, axis=0).tolist()
        return LogSet(summary_data)

    def save(self, filename):
        json.dump(self.data, open(filename, "w+"), indent=2)

    def to_csv(self, filename):
        if not self.data["summary"]:
            raise ValueError("data must be summarized.")
        headers = ["xaxis"]
        datas = [self.data["xaxis"]]
        for t in self.data["values"]:
            for c in self.data["values"][t]:
                headers.append("%s.%s" % (t, c))
                datas.append(self.data["values"][t][c])
        datas = np.array(datas).T
        np.savetxt(filename, datas, header=", ".join(headers))

    def plot(self, settings, filename=None, step=1):
        if not self.data["summary"]:
            raise ValueError("data must be summarized.")
        mpl.rcParams.update(settings["rcParams"])
        fig, axes = plt.subplots(**settings["subplots_args"])
        fig.subplots_adjust(**settings["subplots_adjust_args"])
        axes = np.array(axes).reshape(-1)

        for ax, plot in zip(axes, settings["plots"]):
            ax.grid(True)
            ax.set_title(plot["title"])
            ax.set_xlabel(plot["xlabel"])
            ax.set_ylabel(plot["ylabel"])
            ax.set_yscale(plot["yscale"])
            for type in settings["data-types"]:
                style = plot["default_style"].copy()
                style["label"] = type["name"]
                if "style" in type:
                    style.update(type["style"])
                ax.plot(
                    self.data["xaxis"][::step],
                    self.data["values"][type["typename"]][plot["column"]][::step],
                    **style,
                )
            ax.legend(**plot["legend_args"])

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
