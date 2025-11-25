"""
Distributed Neural Network Training with Ray

This script demonstrates how to train a simple neural network leveraging the distributed 
power of a Ray cluster with GPUs, using shared NFS storage for model persistence.

Usage:
    python distributed_neural_network_training.py [--ray-address RAY_ADDRESS] [--shared-storage-path PATH]

Example:
    python distributed_neural_network_training.py --ray-address ray://192.168.17.93:30002 --shared-storage-path /shared/models
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import ray


class NeuralNetwork(torch.nn.Module):
    """Enhanced neural network with more layers for better GPU utilization."""
    
    def __init__(self, num_inputs: int, num_outputs: int, hidden_size: int = 128):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # Input layer
            torch.nn.Linear(num_inputs, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            # Hidden layers
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.ReLU(),
            # Output layer
            torch.nn.Linear(hidden_size // 4, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


class ToyDataset(Dataset):
    """Simple dataset class for toy data."""
    
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


def compute_accuracy(model, dataloader, device=None):
    """Compute accuracy of model on given dataloader."""
    model.eval()
    correct = 0.0
    total_examples = 0

    for features, labels in dataloader:
        if device is not None:
            features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


def get_job_status(futures: List, job_name: str) -> Tuple[List, List]:
    """Get status of submitted Ray tasks."""
    print(f"\n📋 {job_name} Job Status:")
    print(f"  - Total tasks submitted: {len(futures)}")
    
    ready_futures = []
    pending_futures = []
    
    for i, future in enumerate(futures):
        try:
            ready, not_ready = ray.wait([future], timeout=0)
            if ready:
                ready_futures.append((i, future))
            else:
                pending_futures.append((i, future))
        except Exception as e:
            print(f"  ⚠️  Error checking status of task {i}: {e}")
    
    print(f"  - Tasks completed: {len(ready_futures)}")
    print(f"  - Tasks pending: {len(pending_futures)}")
    
    return ready_futures, pending_futures


@ray.remote(num_gpus=1)
def train_model_task(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the neural network on GPU with given configuration."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Ray worker initialized on device: {device}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA device: {torch.cuda.get_device_name()}")
        print(f"🔧 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, falling back to CPU")
    
    worker_id = ray.get_runtime_context().get_worker_id()
    node_id = ray.get_runtime_context().get_node_id()
    
    print(f"🔧 [Worker {worker_id[:8]}] Starting neural network training")
    print(f"📍 [Worker {worker_id[:8]}] Running on node: {node_id[:8]}")
    print(f"🎯 [Worker {worker_id[:8]}] Device: {device}")
    
    try:
        # Extract training configuration
        num_epochs = training_config.get('num_epochs', 3)
        learning_rate = training_config.get('learning_rate', 0.5)
        batch_size = training_config.get('batch_size', 2)
        seed = training_config.get('seed', 123)
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Create larger training data for better GPU utilization
        num_samples = training_config.get('num_samples', 1000)
        
        # Generate synthetic data
        torch.manual_seed(seed)
        X_train = torch.randn(num_samples, 2).to(device)
        # Create more complex pattern
        y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).long().to(device)
        
        # Create test data (20% of training size)
        test_size = num_samples // 5
        X_test = torch.randn(test_size, 2).to(device)
        y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).long().to(device)
        
        # Create datasets
        train_ds = ToyDataset(X_train.cpu(), y_train.cpu())
        test_ds = ToyDataset(X_test.cpu(), y_test.cpu())
        
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model and move to GPU
        hidden_size = training_config.get('hidden_size', 128)
        model = NeuralNetwork(num_inputs=2, num_outputs=2, hidden_size=hidden_size).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        print(f"📊 [Worker {worker_id[:8]}] Training config: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
        
        # Training loop
        start_time = time.time()
        training_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                # Move data to GPU
                features, labels = features.to(device), labels.to(device)
                
                # Forward pass
                logits = model(features)
                loss = F.cross_entropy(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # GPU memory monitoring (every 10 batches)
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
                    print(f"🔋 [Worker {worker_id[:8]}] GPU Memory - Used: {gpu_memory_used:.2f}GB, Cached: {gpu_memory_cached:.2f}GB")
                
                # Logging
                if batch_idx % max(1, len(train_loader) // 5) == 0:  # Log 5 times per epoch
                    print(f"🔄 [Worker {worker_id[:8]}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                          f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                          f" | Train Loss: {loss:.4f}")
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            training_losses.append(avg_epoch_loss)
            
            print(f"📈 [Worker {worker_id[:8]}] Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Model evaluation
        model.eval()
        
        # Calculate training accuracy
        train_accuracy = compute_accuracy(model, train_loader, device)
        test_accuracy = compute_accuracy(model, test_loader, device)
        
        # Final model outputs for analysis
        with torch.no_grad():
            train_outputs = model(X_train)
            train_probabilities = torch.softmax(train_outputs, dim=1)
        
        print(f"✅ [Worker {worker_id[:8]}] Training completed successfully!")
        print(f"📊 [Worker {worker_id[:8]}] Training accuracy: {train_accuracy:.4f}")
        print(f"📊 [Worker {worker_id[:8]}] Test accuracy: {test_accuracy:.4f}")
        print(f"⏱️  [Worker {worker_id[:8]}] Training time: {training_time:.2f} seconds")
        
        # Return comprehensive results
        # Move model to CPU before returning state_dict to avoid CUDA serialization issues
        model_cpu = model.cpu()
        model_state_dict_cpu = model_cpu.state_dict()
        
        # Clean up GPU memory before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 [Worker {worker_id[:8]}] GPU memory cache cleared")
        
        return {
            'worker_id': worker_id,
            'node_id': node_id,
            'device': str(device),
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'training_losses': training_losses,
            'final_train_outputs': train_outputs.cpu().numpy().tolist(),
            'final_train_probabilities': train_probabilities.cpu().numpy().tolist(),
            'model_state_dict': model_state_dict_cpu,  # CPU version for safe serialization
            'training_config': training_config,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ [Worker {worker_id[:8]}] Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'worker_id': worker_id,
            'node_id': node_id,
            'device': str(device),
            'error': str(e),
            'training_config': training_config,
            'success': False
        }


@ray.remote(num_gpus=1)
def save_model_to_shared_storage(model_state_dict, model_path: str) -> Dict[str, Any]:
    """Save trained model to shared NFS storage."""
    import torch
    import os
    
    worker_id = ray.get_runtime_context().get_worker_id()
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        torch.save(model_state_dict, model_path)
        print(f"💾 [Worker {worker_id[:8]}] Model saved to: {model_path}")
        
        return {
            'success': True,
            'model_path': model_path,
            'worker_id': worker_id
        }
        
    except Exception as e:
        print(f"❌ [Worker {worker_id[:8]}] Error saving model: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'worker_id': worker_id
        }


def create_training_data():
    """Create toy training and test datasets."""
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])

    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def get_training_configurations():
    """Define different training configurations to test in parallel."""
    return [
        {
            'name': 'Standard Training',
            'num_epochs': 20,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_samples': 5000,
            'hidden_size': 256,
            'seed': 123
        },
        {
            'name': 'High Capacity Model',
            'num_epochs': 15,
            'learning_rate': 0.0005,
            'batch_size': 128,
            'num_samples': 8000,
            'hidden_size': 512,
            'seed': 456
        },
        {
            'name': 'Fast Training',
            'num_epochs': 30,
            'learning_rate': 0.002,
            'batch_size': 32,
            'num_samples': 3000,
            'hidden_size': 128,
            'seed': 789
        }
    ]


def connect_to_ray_cluster(ray_address: str):
    """Initialize connection to Ray cluster."""
    print("🚀 Connecting to Ray cluster...")

    # Check CUDA availability on local machine
    print(f"🔍 Local CUDA availability: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 Local CUDA device: {torch.cuda.get_device_name()}")

    # Define runtime environment with required packages
    runtime_env = {
        "pip": [
            "torch==2.4.1",
            "pandas", 
            "numpy"
        ]
    }

    # Connect to Ray cluster
    ray.init(
        address=ray_address,
        runtime_env=runtime_env,
        ignore_reinit_error=True
    )

    print(f"📊 Ray cluster information:")
    print(f"  - Nodes: {len(ray.nodes())}")
    print(f"  - CPUs: {ray.cluster_resources().get('CPU', 0)}")
    print(f"  - Memory: {ray.cluster_resources().get('memory', 0) / (1024**3):.2f} GB")
    print(f"  - GPUs: {ray.cluster_resources().get('GPU', 0)}")

    print("✅ Successfully connected to Ray cluster!")


def submit_training_jobs(training_configs: List[Dict[str, Any]]):
    """Submit distributed training jobs to Ray cluster."""
    print("🎯 Starting distributed neural network training...")
    print(f"📋 Submitting {len(training_configs)} training jobs to Ray cluster...")

    # Submit training tasks directly (no actors)
    futures = []

    for i, config in enumerate(training_configs):
        print(f"🔄 Submitting job {i+1}: {config['name']}")
        
        # Submit training task directly - GPU resources will be released when task completes
        future = train_model_task.remote(config)
        futures.append(future)

    print(f"✅ Successfully submitted {len(futures)} training jobs to Ray cluster")
    print("⏳ Training jobs are now running in parallel on the cluster...")
    
    return futures


def monitor_and_collect_results(futures: List):
    """Monitor training progress and collect results."""
    print("📊 Monitoring training progress...")

    # Monitor progress while jobs are running
    start_time = time.time()
    completed_jobs = 0
    total_jobs = len(futures)

    while completed_jobs < total_jobs:
        # Check job status
        ready_futures, pending_futures = get_job_status(futures, "Neural Network Training")
        
        new_completed = len(ready_futures)
        if new_completed > completed_jobs:
            completed_jobs = new_completed
            elapsed_time = time.time() - start_time
            print(f"⏱️  Progress: {completed_jobs}/{total_jobs} jobs completed ({(completed_jobs/total_jobs)*100:.1f}%) - Elapsed: {elapsed_time:.1f}s")
        
        if completed_jobs < total_jobs:
            time.sleep(5)  # Wait 5 seconds before checking again

    print(f"\n🎉 All training jobs completed!")

    # Collect all results
    print("📋 Collecting results from all training jobs...")
    results = ray.get(futures)

    total_time = time.time() - start_time
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")

    return results, total_time


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and display training results."""
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]

    print(f"\n📊 Training Results Summary:")
    print(f"  ✅ Successful jobs: {len(successful_results)}")
    print(f"  ❌ Failed jobs: {len(failed_results)}")

    if failed_results:
        print(f"\n❌ Failed Jobs Details:")
        for i, result in enumerate(failed_results):
            print(f"  - Job {i}: {result.get('error', 'Unknown error')}")

    if successful_results:
        print(f"\n📈 Successful Jobs Performance:")
        
        for i, result in enumerate(successful_results):
            config = result['training_config']
            print(f"\n  🔹 Job {i+1}: {config.get('name', 'Unknown')}")
            print(f"     Worker: {result['worker_id'][:8]} on Node: {result['node_id'][:8]}")
            print(f"     Device: {result['device']}")
            print(f"     Training time: {result['training_time']:.2f}s")
            print(f"     Train accuracy: {result['train_accuracy']:.4f}")
            print(f"     Test accuracy: {result['test_accuracy']:.4f}")
            print(f"     Final training loss: {result['training_losses'][-1]:.4f}")
        
        # Find best performing model
        best_result = max(successful_results, key=lambda x: x['test_accuracy'])
        print(f"\n🏆 Best Performing Model:")
        print(f"   Configuration: {best_result['training_config']['name']}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"   Training Time: {best_result['training_time']:.2f}s")
        print(f"   Worker: {best_result['worker_id'][:8]} on Node: {best_result['node_id'][:8]}")

    print(f"\n✨ Distributed training completed successfully!")
    
    return successful_results, failed_results


def save_best_model(successful_results: List[Dict[str, Any]], shared_storage_path: str):
    """Save the best trained model to shared storage."""
    if not successful_results:
        print("⚠️  No successful training results to save")
        return

    print("💾 Saving best model to shared storage...")
    
    # Get the best model
    best_result = max(successful_results, key=lambda x: x['test_accuracy'])
    
    # Define paths on shared storage
    timestamp = int(time.time())
    model_filename = f"best_neural_network_model_{timestamp}.pth"
    model_path = os.path.join(shared_storage_path, model_filename)
    
    # Save model using the Ray task (this will create a new task that releases GPU when done)
    save_future = save_model_to_shared_storage.remote(
        best_result['model_state_dict'], 
        model_path
    )
    
    save_result = ray.get(save_future)
    
    if save_result['success']:
        print(f"✅ Model successfully saved to: {model_path}")
        
        # Save training metadata as well
        metadata = {
            'model_path': model_path,
            'training_config': best_result['training_config'],
            'performance': {
                'train_accuracy': best_result['train_accuracy'],
                'test_accuracy': best_result['test_accuracy'],
                'training_time': best_result['training_time'],
                'training_losses': best_result['training_losses']
            },
            'worker_info': {
                'worker_id': best_result['worker_id'],
                'node_id': best_result['node_id'],
                'device': best_result['device']
            },
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(shared_storage_path, f"best_neural_network_metadata_{timestamp}.json")
        try:
            os.makedirs(shared_storage_path, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"📋 Training metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"⚠️  Could not save metadata locally: {e}")
            print("📋 Metadata:", json.dumps(metadata, indent=2))
        
        # Demonstrate model loading
        demonstrate_model_loading(model_path)
        
    else:
        print(f"❌ Failed to save model: {save_result.get('error', 'Unknown error')}")

    print("\n🎯 Model persistence workflow completed!")


def demonstrate_model_loading(model_path: str):
    """Demonstrate loading the saved model."""
    print("\n🔄 Demonstrating model loading from shared storage...")
    
    try:
        # Load model locally to verify it works
        # Use appropriate map_location based on CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = device if torch.cuda.is_available() else "cpu"
        
        loaded_model = NeuralNetwork(num_inputs=2, num_outputs=2, hidden_size=128)
        loaded_state = torch.load(model_path, map_location=map_location, weights_only=True)
        loaded_model.load_state_dict(loaded_state)
        loaded_model = loaded_model.to(device)
        
        print(f"✅ Model successfully loaded from shared storage on {device}!")
        print("🔍 Model verification:")
        
        # Test the loaded model with some sample data
        test_input = torch.tensor([[-1.0, 3.0], [2.5, -1.2]]).to(device)
        loaded_model.eval()
        with torch.no_grad():
            test_output = loaded_model(test_input)
            test_probs = torch.softmax(test_output, dim=1)
        
        print(f"   Test input: {test_input.cpu().numpy().tolist()}")
        print(f"   Test output: {test_output.cpu().numpy().tolist()}")
        print(f"   Test probabilities: {test_probs.cpu().numpy().tolist()}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()


def print_summary():
    """Print final summary and recommendations."""
    print("\n🎊 Script execution completed!")
    print("\n📋 Summary of what we accomplished:")
    print("   ✅ Connected to remote Ray cluster with GPU support")
    print("   ✅ Distributed neural network training across multiple workers")
    print("   ✅ Parallel execution of different training configurations")
    print("   ✅ Performance monitoring and comparison")
    print("   ✅ Best model identification and selection")
    print("   ✅ Model persistence to shared NFS storage")
    print("   ✅ Model loading verification")

    print("\n🚀 Next steps you could try:")
    print("   - Modify training configurations to experiment with different hyperparameters")
    print("   - Scale up the dataset size for more realistic training scenarios")
    print("   - Implement more complex neural network architectures")
    print("   - Add data preprocessing and augmentation pipelines")
    print("   - Implement distributed hyperparameter tuning with Ray Tune")
    print("   - Add model versioning and experiment tracking")

    print("\n💡 Benefits of this Ray-based approach:")
    print("   🔄 Automatic parallelization across available GPUs")
    print("   📈 Linear scaling with cluster size")
    print("   🛡️  Fault tolerance and automatic error handling") 
    print("   📊 Built-in monitoring and logging")
    print("   💾 Seamless integration with shared storage")
    print("   🔧 Easy configuration and deployment")

    print("\n🎯 This script demonstrated how to effectively leverage Ray for distributed deep learning!")


def main():
    """Main function to run distributed neural network training."""
    parser = argparse.ArgumentParser(description='Distributed Neural Network Training with Ray')
    parser.add_argument('--ray-address', 
                       default='ray://192.168.17.93:30002',
                       help='Ray cluster address (default: ray://192.168.17.93:30002)')
    parser.add_argument('--shared-storage-path', 
                       default='/shared/models',
                       help='Path to shared storage for model persistence (default: /shared/models)')
    
    args = parser.parse_args()
    
    try:
        # 1. Connect to Ray cluster
        connect_to_ray_cluster(args.ray_address)
        
        # 2. Create training data
        train_loader, test_loader = create_training_data()
        print("📊 Training and test data created")
        
        # 3. Get training configurations
        training_configs = get_training_configurations()
        
        # 4. Submit training jobs
        futures = submit_training_jobs(training_configs)
        
        # 5. Monitor and collect results
        results, total_time = monitor_and_collect_results(futures)
        
        # 6. Analyze results
        successful_results, failed_results = analyze_results(results)
        
        # 7. Save best model
        save_best_model(successful_results, args.shared_storage_path)
        
        # 8. Print summary
        print_summary()
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up resources...")
        ray.shutdown()
        print("✅ Ray connection closed")


if __name__ == "__main__":
    main()
