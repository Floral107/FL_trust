import argparse
import re
import os
import sys

def main(args):
    set_num_clients(args.clients, args.data)
    print(f"Launching FL {args.data} with {args.clients} clients")

def set_num_clients(num_clients, data):
    """Patch num-supernodes in <data>/pyproject.toml using raw regex
    substitutions so the file structure is never rewritten by toml.dump().

    toml.dump() quotes hyphenated keys (local-simulation → "local-simulation")
    and restructures table headers, both of which break Flower's config reader.
    Targeted regexes on the raw text avoid all of that.

    Also ensures strategy and proximal-mu exist under [tool.flwr.app.config]
    so that --run-config overrides never raise "Key not present" errors on
    servers where these keys were missing from the original pyproject.toml.
    """
    path = os.path.join(data, "pyproject.toml")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # --- 1. patch num-supernodes ---
    new_content, n = re.subn(
        r"(num-supernodes\s*=\s*)\d+",
        rf"\g<1>{num_clients}",
        content,
    )
    if n == 0:
        print(f"[set_num_cl] num-supernodes not found in {path}, appending section.")
        new_content = content.rstrip() + (
            f"\n\n[tool.flwr.federations.local-simulation]\n"
            f"options.num-supernodes = {num_clients}\n"
        )

    # --- 2. ensure strategy key exists under [tool.flwr.app.config] ---
    if not re.search(r"^\s*strategy\s*=", new_content, re.MULTILINE):
        # Insert right after the [tool.flwr.app.config] header line
        new_content, m = re.subn(
            r"(\[tool\.flwr\.app\.config\])",
            r'\1\nstrategy = "fedavg"',
            new_content,
        )
        if m == 0:
            print(f"[set_num_cl] [tool.flwr.app.config] not found in {path}, appending.")
            new_content = new_content.rstrip() + '\n\n[tool.flwr.app.config]\nstrategy = "fedavg"\nproximal-mu = 0.0\n'

    # --- 3. ensure proximal-mu key exists under [tool.flwr.app.config] ---
    if not re.search(r"^\s*proximal-mu\s*=", new_content, re.MULTILINE):
        new_content = re.sub(
            r'(strategy\s*=\s*"[^"]*")',
            r'\1\nproximal-mu = 0.0',
            new_content,
        )

    # Atomic write: concurrent cells (MAX>1) share this pyproject, and a reader
    # (flwr startup) must never observe a truncated mid-write file. Write to a
    # per-PID temp then os.replace() — rename is atomic, so flwr always reads a
    # complete pyproject (old or new), eliminating the "invalid pyproject" race.
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new_content)
    os.replace(tmp, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--clients", type=int, required=True)
    args = parser.parse_args()
    main(args)