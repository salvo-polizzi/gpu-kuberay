import ray
import time

# --- Connection Details ---
# To run this script locally (outside the Kubernetes cluster), you must 
# port-forward the Ray client service (port 10001) from the head node.
# 1. Apply the cluster: kubectl apply -f ray-cluster.yaml
# 2. Wait for it to be ready.
# 3. Port-forward the service: 
#    kubectl port-forward svc/simple-ray-cluster-head-svc 10001:10001
# 4. Run this script.
#
# If running this script inside the cluster (e.g., in a debug pod), 
# use the service name: ray://simple-ray-cluster-head-svc:10001
RAY_HEAD_ADDRESS = "ray://localhost:10001" 

def main():
    """Initializes Ray and executes a simple remote task."""
    print(f"Attempting to connect to Ray cluster at: {RAY_HEAD_ADDRESS}...")
    
    try:
        ray.init(address=RAY_HEAD_ADDRESS)
        print("Successfully connected to the Ray cluster!")

        # 1. Define a remote function (a Ray Task)
        # The '@ray.remote' decorator allows Ray to distribute this function's 
        # execution to any available worker node.
        @ray.remote
        def compute_sum(n):
            """A simple function to compute the sum of numbers up to n."""
            print(f"Worker received task. Computing sum up to {n}...")
            time.sleep(1) # Simulate some work
            return sum(range(n + 1))

        # 2. Execute the task remotely
        # '.remote()' executes the task on a worker and returns an ObjectRef (future).
        future_result = compute_sum.remote(100)
        
        print("\nSubmitted remote task. Waiting for result...")
        
        # 3. Retrieve the result
        # 'ray.get()' blocks until the result is available.
        result = ray.get(future_result)
        
        print(f"\n--- Results ---")
        print(f"Input N: 100")
        print(f"Expected Sum (1 to 100): 5050")
        print(f"Ray Remote Task Result: {result}")
        
        if result == 5050:
            print("Cluster task execution successful!")
        else:
            print("Cluster returned an unexpected result.")

    except Exception as e:
        print(f"\nAn error occurred while connecting or running the task.")
        print("Please ensure the RayCluster is running and the port-forward is active.")
        print(f"Error details: {e}")
    finally:
        # Shutdown the Ray client connection
        if ray.is_initialized():
            ray.shutdown()
            print("\nRay connection shut down.")

if __name__ == "__main__":
    main()
