class CNN_config:
    def __init__(self):
        self.epochs = 200
        self.trainable_embeddings = True
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001
        self.wd = 0.005
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64


class CNN_att_config:
    def __init__(self):
        self.epochs = 200
        self.trainable_embeddings = True
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001
        self.wd = 0.005
        self.att_dim = 64
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64


class CNN_capsule_config:
    def __init__(self):
        self.epochs = 200
        self.trainable_embeddings = True
        self.D = 16
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001  # 0.05067015259077694
        self.wd = 0.005
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64


class CNN_heinsen_capsule_config:
    def __init__(self):
        # Reuters
        self.epochs = 200
        self.trainable_embeddings = True
        self.D = 16
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001  # 0.05067015259077694
        self.wd = 0.005
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64


class CNN_PCaps_config:
    def __init__(self):
        # Reuters
        self.epochs = 200
        self.trainable_embeddings = True
        self.C1 = 1
        self.D = 16
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001  # 0.05067015259077694
        self.wd = 0.005
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64


class CNN_DSA_config:
    def __init__(self):
        # Reuters
        self.epochs = 200
        self.trainable_embeddings = True
        self.C1 = 1
        self.D = 16
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001  # 0.05067015259077694
        self.wd = 0.005
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64


class CNN_custom_config:
    def __init__(self):
        # Reuters
        self.epochs = 200
        self.trainable_embeddings = True
        self.C1 = 1
        self.D = 16
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001  # 0.05067015259077694
        self.wd = 0.005
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64


class CNN_custom2_config:
    def __init__(self):
        # Reuters
        self.epochs = 200
        self.trainable_embeddings = True
        self.C1 = 1
        self.D = 16
        self.embedding_dropout = 0
        self.output_channel = 128
        self.dropout = 0.2
        self.lr = 0.001
        self.wd = 0.005
        self.total_batch_size = 64
        self.train_batch_size = 64
        self.val_batch_size = 64
