import ray
import torch
import torch.nn as nn
import torch.optim as optim
import time

# --------------------------------------------------------------------
# Cluster connection
# --------------------------------------------------------------------
# If you are running locally, port-forward first:
#   kubectl port-forward svc/simple-ray-cluster-head-svc 10001:10001
# Then use: "ray://localhost:10001"
# Inside the cluster, use the Ray service name:
#   "ray://simple-ray-cluster-head-svc:10001"
RAY_HEAD_ADDRESS = "ray://172.29.5.95:30001"


# --------------------------------------------------------------------
# Define a simple PyTorch model and training function
# --------------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, output_size=1):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


@ray.remote
def train_worker(worker_id: int, epochs: int = 3, lr: float = 0.01):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    """Each Ray task trains a simple model on random data."""
    print(f"[Worker {worker_id}] Starting training...")
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Fake dataset (random input-output pairs)
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"[Worker {worker_id}] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        time.sleep(1)

    final_loss = loss.item()
    print(f"[Worker {worker_id}] Training complete. Final loss: {final_loss:.4f}")
    return {"worker_id": worker_id, "final_loss": final_loss}


# --------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------
def main():
    print(f"Connecting to Ray cluster at {RAY_HEAD_ADDRESS}...")

    # Define runtime environment with required packages
    runtime_env = {
        "pip": [
            "torch==2.2.2",
            "pandas",
            "numpy"
        ]
    }

    ray.init(address=RAY_HEAD_ADDRESS)
    print("Connected successfully!")

    # Launch 2 remote training tasks
    workers = [train_worker.remote(i) for i in range(2)]

    print("\nWaiting for training results...")
    results = ray.get(workers)

    print("\n--- Training Summary ---")
    for r in results:
        print(f"Worker {r['worker_id']} -> Final Loss: {r['final_loss']:.4f}")

    print("\nAll remote tasks completed successfully!")
    ray.shutdown()


if __name__ == "__main__":
    main()

