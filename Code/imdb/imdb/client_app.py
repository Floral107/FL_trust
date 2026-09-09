"""imdb: A Flower / TensorFlow app."""

import gc
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from imdb.task import load_data, load_model, process_text, get_num_clients, save_client, train_with_proximal
import numpy as np

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
        proximal_mu = float(config.get("proximal_mu", 0.0))
        strategy = "fedprox" if proximal_mu > 0.0 else "fedavg"

        if proximal_mu > 0.0:
            train_with_proximal(
                self.model, self.x_train, self.y_train,
                mu=proximal_mu,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                shuffle=False,
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
        num_clients = get_num_clients()
        # Save model
        save_client(self.model, num_clients, self.partition_id, round_num, strategy=strategy)
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
