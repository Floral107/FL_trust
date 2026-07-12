"""imdbnoniid: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from imdbnoniid.task import (
    load_data, load_model, process_text, get_num_clients, save_client,
    train_with_proximal, train_with_adv, train_with_dpsgd,
)


import gc
import logging
import ray

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose, partition_id
    ):
        self.model = model
        x_train_texts, y_train, x_test_texts, y_test = data
        # Embed in chunks (see process_text); convert to numpy so TF graph refs
        # are released and the raw text strings can be GC'd immediately.
        self.x_train = process_text(x_train_texts)
        del x_train_texts
        gc.collect()
        self.x_test = process_text(x_test_texts)
        del x_test_texts
        gc.collect()
        self.y_train = y_train
        self.y_test = y_test
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
            # IMDB has no sensitive attribute; treat fair as a no-op (BL training).
            self.model.fit(
                self.x_train, self.y_train,
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose,
            )
        elif et_mode == "adv":
            train_with_adv(
                self.model, self.x_train, self.y_train,
                ratio=float(config.get("et_adv_ratio", 0.5)),
                eps=float(config.get("et_adv_eps", 0.0015)),
                alpha=float(config.get("et_adv_alpha", 0.0015)),
                pgd_steps=int(config.get("et_adv_steps", 7)),  # dead default: the server's
                # on_fit_config ALWAYS supplies et_adv_steps (=7); aligned to avoid confusion.
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, shuffle=False,
                # USE embeddings: no natural value clip → use wide bounds.
                clip_min=-1e6, clip_max=1e6,
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
            )

        round_num = config.get("server_round", 0)

        # Save model
        save_client(model=self.model, client_num=get_num_clients(),
                    partition_id=self.partition_id, round_num=round_num,
                    strategy=strategy, et_mode=et_mode,
                    weight_mode=weight_mode, weight_metric=weight_metric)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose, partition_id=partition_id
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
