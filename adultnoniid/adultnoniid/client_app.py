"""adultnoniid: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from adultnoniid.task import (
    load_data, load_model, get_num_clients, save_client, set_seed, SEED,
    train_with_proximal, train_with_fair, train_with_adv, train_with_dpsgd,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose, partition_id
    ):
        self.model = model
        # data may carry s_train (when et_mode=fair is requested)
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
        set_seed(SEED)  # Set the seed for reproducibility
        self.model.set_weights(parameters)

        et_mode = str(config.get("et_mode", "none")).lower()
        proximal_mu = float(config.get("proximal_mu", 0.0))
        # ST/DY weight-mode reaches client only so client ckpts land in
        # 420_st_<metric>/.../ instead of 420/.../ (purely for save path).
        weight_mode = str(config.get("weight_mode", "none")).lower()
        weight_metric = str(config.get("weight_metric", "")).lower()

        # ET takes precedence: BL/ET strategy label only affects save path.
        if et_mode in ("fair", "adv", "dp"):
            strategy = "fedavg"
        elif proximal_mu > 0.0:
            strategy = "fedprox"
        else:
            strategy = "fedavg"

        if et_mode == "fair":
            if self.s_train is None:
                raise RuntimeError("et_mode=fair requires sensitive attributes; load_data must be called with return_sensitive=True")
            train_with_fair(
                self.model, self.x_train, self.y_train, self.s_train,
                lambda_fair=float(config.get("et_lambda_fair", 0.1)),
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, shuffle=False,
            )
        elif et_mode == "adv":
            train_with_adv(
                self.model, self.x_train, self.y_train,
                ratio=float(config.get("et_adv_ratio", 0.5)),
                eps=float(config.get("et_adv_eps", 0.3)),
                alpha=float(config.get("et_adv_alpha", 0.007)),
                pgd_steps=int(config.get("et_adv_steps", 7)),
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, shuffle=False,
                # ADULT features are standardised / one-hot floats → no value clipping.
                clip_min=-1e6, clip_max=1e6,
                loss_name="binary_crossentropy",
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
                self.x_train, self.y_train,
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose,
            )

        round_num = config.get("server_round", 0)
        save_client(
            model=self.model, client_num=get_num_clients(),
            partition_id=self.partition_id, round_num=round_num,
            strategy=strategy, et_mode=et_mode,
            weight_mode=weight_mode, weight_metric=weight_metric,
        )
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
    et_mode = str(context.run_config.get("et-mode", "none")).lower()
    # Need sensitive attribute only for fair ET; otherwise skip to save memory.
    needs_sensitive = (et_mode == "fair")
    data = load_data(partition_id, num_partitions, return_sensitive=needs_sensitive)
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
