"""celebanoniid: A Flower / TensorFlow app."""
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from celebanoniid.task import (
    load_data, load_model, get_num_clients, save_client, set_seed, SEED,
    train_with_proximal, train_with_fair, train_with_adv, train_with_dpsgd,
)


set_seed(SEED)


class FlowerClient(NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose, partition_id):
        self.model = model
        if len(data) == 6:
            self.x_train, self.y_train, self.x_test, self.y_test, self.s_train, self.s_test = data
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = data
            self.s_train = self.s_test = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.partition_id = partition_id

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        et_mode = str(config.get("et_mode", "none")).lower()
        proximal_mu = float(config.get("proximal_mu", 0.0))
        weight_mode = str(config.get("weight_mode", "none")).lower()
        weight_metric = str(config.get("weight_metric", "")).lower()

        if et_mode in ("fair", "adv", "dp"):
            strategy = "fedavg"
        elif proximal_mu > 0.0:
            strategy = "fedprox"
        else:
            strategy = "fedavg"

        if et_mode == "fair":
            if self.s_train is None:
                raise RuntimeError("et_mode=fair requires sensitive attributes")
            train_with_fair(
                self.model, self.x_train, self.y_train, self.s_train,
                lambda_fair=float(config.get("et_lambda_fair", 0.1)),
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, shuffle=False,
                # CelebA: softmax 2-class → P(ŷ=1) is column 1, sensitive "Male" = 1.
                sensitive_pos=1, target_pos=1,
            )
        elif et_mode == "adv":
            train_with_adv(
                self.model, self.x_train, self.y_train,
                ratio=float(config.get("et_adv_ratio", 0.5)),
                eps=float(config.get("et_adv_eps", 0.03)),
                alpha=float(config.get("et_adv_alpha", 0.007)),
                pgd_steps=int(config.get("et_adv_steps", 7)),
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, shuffle=False,
                # Images normalized to [0, 1].
                clip_min=0.0, clip_max=1.0,
                loss_name="sparse_categorical_crossentropy",
            )
        elif et_mode == "dp":
            train_with_dpsgd(
                self.model, self.x_train, self.y_train,
                clip_norm=float(config.get("et_dp_clip", 1.0)),
                noise_multiplier=float(config.get("et_dp_sigma", 1.0)),
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, shuffle=False,
            )
        elif proximal_mu > 0.0:
            train_with_proximal(
                self.model, self.x_train, self.y_train,
                mu=proximal_mu,
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, shuffle=False,
            )
        else:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                shuffle=False,
            )

        round_num = config.get("server_round", 0)
        client_num = get_num_clients()

        save_client(self.model, client_num, self.partition_id, round_num,
                    strategy=strategy, et_mode=et_mode,
                    weight_mode=weight_mode, weight_metric=weight_metric)

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    set_seed(SEED)
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    et_mode = str(context.run_config.get("et-mode", "none")).lower()
    needs_sensitive = (et_mode == "fair")
    data = load_data(partition_id, num_partitions, return_sensitive=needs_sensitive)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    return FlowerClient(
        net, data, epochs, batch_size, verbose, partition_id=partition_id
    ).to_client()


app = ClientApp(client_fn=client_fn)
