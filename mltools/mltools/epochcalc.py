# -*- coding:utf-8 -*-


class EpochCalc:
    def __init__(self, train_epoch, data_size, minibatch_size):
        self.train_epoch = train_epoch
        self.train_update = data_size / minibatch_size * train_epoch
        self.data_size = data_size
        self.minibatch_size = minibatch_size

        if not self.train_update.is_integer():
            raise ValueError(
                "train_update must be integer: ({}) ".format(self.train_update)
            )
        self.train_update = int(self.train_update)

    def update_to_epoch(self, update_time, force_integer=True):
        epoch = update_time * self.minibatch_size / self.data_size
        if force_integer and not epoch.is_integer():
            raise ValueError("epoch should be integer, but got {}".format(epoch))
        epoch = int(epoch) if force_integer else epoch
        return epoch

    def epoch_to_update(self, epoch, force_integer=True):
        update = self.data_size / self.minibatch_size * epoch
        if force_integer and not update.is_integer():
            raise ValueError("update should be integer, but got {}".format(update))
        update = int(update) if force_integer else update
        return update

    def update_per_epoch(self, **kwargs):
        return self.epoch_to_update(1, **kwargs)

    def epoch_per_update(self, **kwargs):
        return self.update_to_epoch(1, **kwargs)
