# GPU Ray Workspaces on Kubernetes

This repository contains the Git-tracked Helm charts and example scripts for a GPU-oriented, multi-user development and execution environment based on:

- Kubernetes
- Helm
- Argo CD
- KubeRay / RayJob / RayCluster
- `code-server` (VS Code in the browser)
- PVC-backed shared storage

The project evolved through multiple deployment models:

- a long-lived RayCluster shared with the workspace
- a per-workspace long-lived RayCluster
- a per-workspace RayJob that creates an ephemeral RayCluster on demand

This README intentionally documents only the files and directories that are currently tracked by Git in this repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`/Users/salvopolizzi/uni/gpu-kuberay/nfs-system`](./nfs-system) | Helm chart for a simple in-cluster NFS server. |
| [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray`](./vscode-ray) | Helm chart for `code-server` + dynamic PVC + long-lived per-workspace RayCluster. |
| [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob`](./vscode-rayjob) | Helm chart for `code-server` + dynamic PVC + RayJob submission workflow that creates an ephemeral RayCluster. |
| [`/Users/salvopolizzi/uni/gpu-kuberay/scripts`](./scripts) | Standalone Ray/PyTorch example scripts used for testing and demonstrations. |

## Prerequisites

Before deploying the charts, make sure the cluster already has:

- Kubernetes with GPU-capable nodes if you want GPU workers
- Helm
- Argo CD if you deploy through GitOps
- KubeRay operator installed
- NVIDIA device plugin and runtime class if using GPU workers
- a storage backend:
  - dynamic RWX storage class such as `nfs-client` for `vscode-ray` and `vscode-rayjob`
  - or a static PV target for `vscode-rayjob-static`

## Folder-by-folder usage

### `/nfs-system`

This chart deploys a simple NFS server inside the cluster.

Main files:

- [`/Users/salvopolizzi/uni/gpu-kuberay/nfs-system/Chart.yaml`](/Users/salvopolizzi/uni/gpu-kuberay/nfs-system/Chart.yaml)
- [`/Users/salvopolizzi/uni/gpu-kuberay/nfs-system/values.yaml`](/Users/salvopolizzi/uni/gpu-kuberay/nfs-system/values.yaml)

What it creates:

- namespace
- NFS server `Deployment`
- NFS `Service`

Key values to customize:

- `nfs.namespace.name`: namespace for the NFS server
- `nfs.server.hostPath.path`: host path used by the NFS container
- `nfs.server.service.clusterIP`: fixed cluster IP if you want a stable internal address
- `nfs.server.replicas`: usually keep at `1`

Example deployment:

```bash
helm upgrade --install nfs-system /Users/salvopolizzi/uni/gpu-kuberay/nfs-system
```

Use this only if you want the repository to also manage the backing NFS server itself. If your cluster already has an external NFS server and a provisioner, this chart is optional.

### `/scripts`

This folder contains standalone Ray examples used during development.

Main files:

- [`/Users/salvopolizzi/uni/gpu-kuberay/scripts/distributed_network_train.py`](/Users/salvopolizzi/uni/gpu-kuberay/scripts/distributed_network_train.py)
- [`/Users/salvopolizzi/uni/gpu-kuberay/scripts/simple_training.py`](/Users/salvopolizzi/uni/gpu-kuberay/scripts/simple_training.py)
- [`/Users/salvopolizzi/uni/gpu-kuberay/scripts/sum.py`](/Users/salvopolizzi/uni/gpu-kuberay/scripts/sum.py)

These are not Helm charts. They are useful as:

- prototypes
- smoke tests
- examples to adapt into `entrypoint.py` or other workspace scripts

### `/vscode-ray`

This chart deploys one browser workspace plus one long-lived RayCluster for that workspace.

Main files:

- [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray/Chart.yaml`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray/Chart.yaml)
- [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray/values.yaml`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray/values.yaml)

What it creates:

- `code-server` `Deployment`
- `Service`
- optional `Ingress`
- dynamic `PVC`
- long-lived `RayCluster`

Use this chart when:

- you want a persistent RayCluster attached to the workspace
- you want to connect directly from the workspace to a stable Ray head service

Important values to customize:

- `workspace.image.repository`
- `workspace.image.tag`
- `service.type` and optionally `service.nodePort`
- `ingress.enabled`, `ingress.hosts`, `ingress.className`
- `sharedStorage.storageClassName`
- `sharedStorage.size`
- `rayCluster.head.image.*`
- `rayCluster.workers.cpuGroup.*`
- `rayCluster.workers.gpuGroup.*`
- `extraVolumes`

Example deployment:

```bash
helm upgrade --install vscode-ray /Users/salvopolizzi/uni/gpu-kuberay/vscode-ray
```

Example custom overrides:

```bash
helm upgrade --install vscode-ray /Users/salvopolizzi/uni/gpu-kuberay/vscode-ray \
  --set workspace.image.tag=1.0.6 \
  --set sharedStorage.storageClassName=nfs-client \
  --set rayCluster.workers.gpuGroup.replicas=1
```

### `/vscode-rayjob`

This is the main on-demand execution chart.

It deploys a persistent browser workspace, but Ray compute is created only when the user submits a RayJob from inside the workspace.

Main files:

- [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob/Chart.yaml`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob/Chart.yaml)
- [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob/values.yaml`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob/values.yaml)

What it creates:

- `code-server` `Deployment`
- `Service`
- optional `Ingress`
- dynamic `PVC`
- `ServiceAccount`, `Role`, `RoleBinding`
- mounted RayJob template `ConfigMap`

What it does not create immediately:

- no persistent RayCluster

Instead, the user runs:

```bash
./submit_rayjob.sh
```

from inside the workspace, and that script creates a `RayJob` resource that launches an ephemeral RayCluster.

Important values to customize:

- `workspace.image.repository`
- `workspace.image.tag`
- `workspace.projectPath`
- `service.type` and `service.nodePort`
- `sharedStorage.storageClassName`
- `sharedStorage.size`
- `sharedStorage.projectSubPath`
- `rayJob.entrypoint`
- `rayJob.runtimeEnv`
- `rayJob.rayClusterSpec.head.image.*`
- `rayJob.rayClusterSpec.workers.cpuGroup.*`
- `rayJob.rayClusterSpec.workers.gpuGroup.*`
- `extraVolumes`

Notes:

- The default workflow assumes the editable project is on the shared PVC.
- The RayJob entrypoint points to `/shared/project/entrypoint.py`.
- Runtime dependencies can be supplied through `rayJob.runtimeEnv`, but pre-baking dependencies into the Ray image is usually more reliable for production.

Example deployment:

```bash
helm upgrade --install vscode-rayjob /Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob
```

Useful submission commands inside the workspace:

```bash
./submit_rayjob.sh
./submit_rayjob.sh --profile
./submit_rayjob.sh --entrypoint /shared/project/other_script.py
```

## Choosing the right chart

Use [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray) if:

- you want a stable, always-on RayCluster per workspace
- you are okay paying idle resource cost for lower submission latency

Use [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob) if:

- you want per-workspace persistent storage
- you want Ray compute only when a job is submitted
- you want the most flexible model for per-user PVC reuse

### Expose VS Code through NodePort

```yaml
service:
  type: NodePort
  nodePort: 30081
```

Then browse to:

```text
http://<node-ip>:30081
```

### Expose VS Code through Ingress

```yaml
ingress:
  enabled: true
  className: traefik
  hosts:
    - host: your-host.example.com
      paths:
        - path: /
          pathType: Prefix
```

### Change the storage class

For dynamic PVC-based charts:

```yaml
sharedStorage:
  storageClassName: nfs-client
```

### Mount extra PVC-backed volumes

All workspace charts support `extraVolumes` for additional PVC mounts.

Example:

```yaml
extraVolumes:
  - name: datasets
    mountPath: /data
    claimName: datasets-pvc
    readOnly: true
  - name: checkpoints
    mountPath: /mnt/checkpoints
    claimName: checkpoints-pvc
```

For `vscode-ray`, these mounts are applied to:

- VS Code pod
- Ray head
- Ray workers

For `vscode-rayjob` and `vscode-rayjob-static`, these mounts are applied to:

- VS Code pod
- RayJob submitter pod
- RayJob ephemeral Ray head
- RayJob ephemeral Ray workers

## Argo CD usage

Each chart is self-contained and can be used directly as an Argo CD application source path.

Typical `spec.source.path` values:

- `nfs-system`
- `vscode-ray`
- `vscode-rayjob`

Typical deployment model:

1. Deploy storage prerequisites first.
2. Deploy one workspace chart per workspace or per user.
3. Override values per application using:
   - `values.yaml` in Git
   - or `spec.source.helm.values`

Example `Application` source snippet:

```yaml
source:
  repoURL: <your-repo-url>
  targetRevision: main
  path: vscode-rayjob
  helm:
    values: |
      workspace:
        image:
          tag: 1.0.6
      sharedStorage:
        storageClassName: nfs-client
      service:
        nodePort: 30081
```

## Suggested deployment order

If you want the full stack from this repository:

1. Deploy [`/Users/salvopolizzi/uni/gpu-kuberay/nfs-system`](/Users/salvopolizzi/uni/gpu-kuberay/nfs-system) if you need the in-cluster NFS server.
2. Install or verify the NFS provisioner if you use dynamic PVCs.
3. Build and push the workspace image from [`/Users/salvopolizzi/uni/gpu-kuberay/vscode_image`](/Users/salvopolizzi/uni/gpu-kuberay/vscode_image).
4. Deploy one of:
   - [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-ray)
   - [`/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob`](/Users/salvopolizzi/uni/gpu-kuberay/vscode-rayjob)

## Notes and caveats

- `helm` templates were not validated inside this repository environment automatically; validate in your cluster or CI before production use.
- The `vscode-rayjob` flow is the most flexible variant for per-user storage reuse, but it can have noticeable cold-start latency because it provisions compute on demand.
- Runtime `pip` installation inside `RayJob.runtimeEnv` can cause slow startup or failures if network access to PyPI is unstable. Baking dependencies into the Ray image is often a better production choice.
- There may be local, untracked files or directories in the working tree, but they are intentionally not described here because they are not part of the tracked repository state.
