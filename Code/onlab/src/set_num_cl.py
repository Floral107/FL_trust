import argparse
import toml
import os

def main(args):
    set_num_clients(args.clients, args.data)

    # You'd probably want to import and run your FLWR logic here
    print(f"Launching FL {args.data} with {args.clients} clients" )
    
def set_num_clients(num_clients, data):
    
    path = os.path.join("/home/ulrich", data, "pyproject.toml")
    config = toml.load(path)

    config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"] = num_clients

    with open(path, "w") as f:
        toml.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--clients", type=int, required=True)
    args = parser.parse_args()
    main(args)