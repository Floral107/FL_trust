"""cifar: A Flower / TensorFlow app."""
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from cifar.task import load_data, load_model, get_num_clients, save_client, set_seed, SEED, train_with_proximal



set_seed(SEED)  # Set seed for reproducibility

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose, partition_id
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.partition_id = partition_id


    def fit(self, parameters, config):
        set_seed(SEED)
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
                shuffle=False,
            )

        round_num = config.get("server_round", 0)
        client_num = get_num_clients()

        save_client(self.model, client_num, self.partition_id, round_num, strategy=strategy)

        return self.model.get_weights(), len(self.x_train), {}
    

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accu.racy": accuracy}


def client_fn(context: Context):
    set_seed(SEED)  # Set seed for reproducibility
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
