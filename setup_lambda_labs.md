# Cloud Computing Setup Guide (Lambda Labs)

This guide provides instructions for setting up cloud GPU computing resources on **Lambda Labs** to run our robust overfitting experiments.

---

## 1. Provisioning a Cloud GPU Instance

For PreActResNet-18 training runs on CIFAR-10, we recommend using an **NVIDIA A10 (24 GB PCIe)** instance on Lambda Labs, which balances compute speed and cost (~$0.75/hour).

1. Log in or create an account at [Lambda Labs](https://lambdalabs.com/).
2. Navigate to **SSH Keys** tab and upload your local machine's public SSH key (typically located at `~/.ssh/id_rsa.pub` or `~/.ssh/id_ed25519.pub`).
3. Click **Launch Instance** in dashboard.
4. Follow launch instance wizard:
   * **Instance type:** Select **A10 (24 GB PCIe)** instance type.
   * **Region:** Choose an available region.
   * **Base image:** Select **Lambda Stack 24.04** (comes pre-configured with CUDA and PyTorch).
   * **Filesystem:** Keep default settings (**No filesystem**, as we will use fast local instance storage).
   * **Security:** Select **Global firewall rules** and click **Confirm** (Lambda Labs will automatically associate your uploaded SSH key).
5. Click final launch button.
6. Once status shows `Running`, note instance IP address (`<INSTANCE_IP>`).

---

## 2. Accessing Instance via SSH

Open your local terminal and connect to instance:

```bash
ssh ubuntu@<INSTANCE_IP>
```

To configure SSH for easy access, add an alias to your local configuration (`~/.ssh/config`):

```text
Host lambda-overfit
    HostName <INSTANCE_IP>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519  # Adjust to your private key path
```

Then you can connect by simply typing:
```bash
ssh lambda-overfit
```

---

## 3. Synchronizing Code to Instance

You can sync your local workspace to cloud GPU instance using `rsync` (recommended) or via Git.

### Option A: Using `rsync` (Fastest for iteration)
Run this command from your local workspace root directory:

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'cifar-data' --exclude 'cifar_model' ./ lambda-overfit:~/Robust-Overfitting/
```

### Option B: Using Git
On your cloud instance, clone public repository:

```bash
git clone https://github.com/<USERNAME>/Robust-Overfitting.git
cd Robust-Overfitting
```

---

## 4. Remote Environment Configuration

Lambda Labs instances come pre-configured with CUDA and PyTorch, but we recommend creating a project-specific virtual environment:

```bash
# SSH into machine
ssh lambda-overfit
cd ~/Robust-Overfitting

# Create a virtual environment using system Python
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install project dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Running Long-Running Jobs with `tmux`

Since adversarial training runs can take several hours, SSH connection drops will terminate training script unless handled properly. Use `tmux` to run jobs in background:

1. Start a new `tmux` session named `training`:
   ```bash
   tmux new -s training
   ```
2. Activate your virtual environment and start training (example script to be created in Week 3):
   ```bash
   source .venv/bin/activate
   python train_cifar.py --fname runs/baseline_run
   ```
3. Detach from session by pressing `Ctrl + B`, then `D`.
4. You can now close your terminal or disconnect.
5. To re-attach and check progress, SSH back in and run:
   ```bash
   tmux attach -t training
   ```

---

## 6. Remote TensorBoard Monitoring

To monitor training curves (like train/test robust loss and accuracy) in real-time, you can use SSH port forwarding.

1. Start TensorBoard on cloud instance pointing to logs directory:
   ```bash
   tensorboard --logdir=runs/ --port=6006
   ```
2. On your local machine, open a terminal and run forwarding command:
   ```bash
   ssh -N -L 6006:localhost:6006 lambda-overfit
   ```
3. Open [http://localhost:6006](http://localhost:6006) in your local web browser to view interactive training plots.

---

## 7. Terminating Instance to Stop Billing

> [!CAUTION]
> Lambda Labs instances are billed continuously while they are in **Booting** or **Running** state. Shutting down OS from within terminal (e.g., `sudo shutdown`) **does not** stop billing.

To stop incurring charges, you must terminate instance through dashboard:
1. Go to Lambda Labs dashboard under **Instances** tab.
2. Select checkbox next to your running instance.
3. Click **Terminate** button in top-right corner of dashboard.

> [!WARNING]
> Terminating an instance deletes all files and data stored on its local storage. Be sure to pull/synchronize any training logs, model checkpoints, or results to your local machine (using `rsync` or Git) **before** terminating.
