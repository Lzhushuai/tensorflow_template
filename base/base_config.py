from bunch import Bunch


class Config(Bunch):
    def __init__(self, name,
                 n_feature: list, n_class: int,
                 n_batch=64, n_step=100, n_epoch=20, learning_rate=0.001,
                 ckpt_dir=None, summary_dir=None,
                 **kwargs):
        super(Config, self).__init__()

        self.name = name
        self.n_feature = n_feature
        self.n_class = n_class

        self.n_batch = n_batch
        self.n_step = n_step
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.ckpt_dir = ckpt_dir
        self.summary_dir = summary_dir

        for k, v in kwargs.items():
            self[k] = v


if __name__ == '__main__':
    config = Config('', [4], 3, n_batch=11, aaa="AAA")
    config.ttttt = 'ttttt'

    print(config.n_batch)  # 11
    print(config.aaa)  # AAA

    print(config.ttttt)  # ttttt
    print(config['ttttt'])  # ttttt