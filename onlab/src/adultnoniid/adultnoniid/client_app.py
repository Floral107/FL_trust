"""adultnoniid: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from adultnoniid.task import load_data, load_model, get_num_clients, save_client, set_seed, SEED

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
        set_seed(SEED)  # Set the seed for reproducibility
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        round_num = config.get("server_round", 0)
        
        # Save model
        save_client(model=self.model, client_num=get_num_clients(), partition_id=self.partition_id, round_num=round_num)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_seed(SEED)
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    set_seed(SEED)
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
