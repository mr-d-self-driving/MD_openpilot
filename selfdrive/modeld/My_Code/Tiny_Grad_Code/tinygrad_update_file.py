# find_trainable.py  ── run just once, writes trainable.txt
import onnx, json, argparse
from collections import deque, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--sources", nargs="+",
                    default=["nav_features", "nav_instructions"])
args = parser.parse_args()

m      = onnx.load("/home/adas/openpilot/openpilot/selfdrive/modeld/models/driving_policy_with_normal_nav.onnx")
G      = m.graph
inputs = set(args.sources)                    # ⇠ BFS frontier
reach  = set(inputs)                          # ⇠ everything reachable
Q      = deque(inputs)

# --- make a quick index: tensor  → nodes that consume it
consumers = defaultdict(list)
for node in G.node:
    for x in node.input:
        consumers[x].append(node)

while Q:
    t = Q.popleft()
    for node in consumers.get(t, []):
        # if any input already reachable, outputs become reachable
        if any(i in reach for i in node.input):
            for o in node.output:
                if o not in reach:
                    reach.add(o)
                    Q.append(o)

# Everything in reach now *depends* on nav_* inputs.
init_name_set = {init.name for init in G.initializer}
trainable     = set()

for node in G.node:
    if any(i in reach for i in node.input):   # downstream node
        for i in node.input:                  # keep its weights trainable
            if i in init_name_set:
                trainable.add(i)

print(f"{len(trainable)} weights downstream of nav_*")
with open("trainable.txt", "w") as f:
    f.write("\n".join(sorted(trainable)))
