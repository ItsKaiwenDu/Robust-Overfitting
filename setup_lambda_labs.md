# Cloud Computing Setup Guide (Lambda Labs)

This guide provides instructions for setting up cloud GPU computing resources on **Lambda Labs** to run our robust overfitting experiments.

> **Note:** This guide is written for **macOS** local machines.

*Last updated: July 27, 2026.*

---

## Prerequisites: Generate an SSH Key

Before launching an instance, you need an SSH key pair on your local machine. To check if you already have one:

```bash
ls ~/.ssh/id_ed25519.pub
```

If the file exists, you are good to go. Skip to Step 1. If not, follow instruction below:

```bash
ssh-keygen -t ed25519
```

Press **Enter** to accept default file location and optionally set a passphrase. Then copy the public key to paste into Lambda Labs later:

```bash
cat ~/.ssh/id_ed25519.pub
```

---

## 1. Provisioning a Cloud GPU Instance

For PreActResNet-18 training runs on CIFAR-10, we use **NVIDIA A10 (24 GB PCIe)** instance on Lambda Labs, which balances compute speed and cost ($1.29/hour).

1. Log in or create an account at [Lambda Labs](https://lambdalabs.com/).
2. Navigate to **SSH Keys** tab and upload your local machine's public SSH key (typically located at `~/.ssh/id_rsa.pub` or `~/.ssh/id_ed25519.pub`).
3. Click **Launch Instance** in dashboard.
4. Follow launch instance wizard:
   * **Instance type:** Select **A10 (24 GB PCIe)** instance type.
   * **Region:** Choose an available region.
   * **Base image:** Select **Lambda Stack 24.04** (comes pre-configured with CUDA and PyTorch).
   * **Filesystem:** Select **Don't attach a filesystem**, as we will use fast local instance storage.
   * **Security:** Select **Global firewall rules** and click **Confirm** (Lambda Labs will automatically associate your uploaded SSH key).
5. Click **Confirm** to launch the instance.
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

Sync your local workspace to the cloud GPU instance using `rsync`. Run this command from your local workspace root directory:

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'cifar-data' --exclude 'cifar_model' ./ lambda-overfit:~/Robust-Overfitting/
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
2. Activate your virtual environment and start training:
   ```bash
   source .venv/bin/activate
   python scripts/train.py
   ```
   Running with no flags uses our project's defaults: PreActResNet-18 on CIFAR-10, 200 epochs, PGD-10 adversarial training (epsilon = 8/255, step size = 2/255), SGD with momentum 0.9 and weight decay 5e-4, learning rate decaying at epochs 100 and 150. A checkpoint is saved and a test-set evaluation runs every 5 epochs (plus epoch 1 and the final epoch). Training progress and the same clean/robust metrics are also logged to TensorBoard as it runs (see Section 6). Add `--diagnostic` to instead run a quick 1-epoch sanity check on 10% of the data before committing to a full run.
3. Detach from session by pressing `Ctrl + B`, then `D`.
4. You can now close your terminal or disconnect.
5. To re-attach and check progress, SSH back in and run:
   ```bash
   tmux attach -t training
   ```

---

## 6. Remote TensorBoard Monitoring

To monitor training curves (like train/test robust loss and accuracy) in real-time, you can use SSH port forwarding.

1. On the cloud instance, open a new tmux session and start TensorBoard:
   ```bash
   tmux new -s tensorboard
   tensorboard --logdir=runs/ --port=6006
   ```
2. On your local machine, open a new terminal and run the port forwarding command:
   ```bash
   ssh -N -L 6006:localhost:6006 lambda-overfit
   ```
3. Open [http://localhost:6006](http://localhost:6006) in your local web browser to view interactive training plots.

---

## 7. Checking Training Progress

After detaching from the tmux session, you can check on the training run at any time by re-attaching to see live terminal output (epoch logs, loss values):
```bash
ssh lambda-overfit
tmux attach -t training
```
Detach again with `Ctrl + B`, then `D`.

For real-time accuracy and loss curves in your browser instead, see Section 6 (TensorBoard).

---

## 8. Running Evaluation (PGD-20)

Once training has finished (or you want to check progress on saved checkpoints so far), run `evaluate.py` to score every saved checkpoint on the CIFAR-10 test set, both on clean images and under PGD-20 adversarial attack:

```bash
ssh lambda-overfit
cd ~/Robust-Overfitting
source .venv/bin/activate
python scripts/evaluate.py
```

This defaults to evaluating every `.pt`/`.pth` file in `checkpoints/` against a 20-step PGD attack (epsilon = 8/255, step size = 2/255), and writes one row per checkpoint to `report/evaluation_results.csv` (columns: `epoch, clean_loss, clean_acc, robust_loss, robust_acc, eval_time_sec`). Results are written after each checkpoint, so partial results are saved even if evaluation is interrupted partway through. Like training, this can take a while, so it's worth running inside its own `tmux` session (see Section 5) rather than the same session as training.

---

## 9. Downloading Results from Instance

Run the following commands **from your local machine** to download results before terminating the instance:

```bash
# Download model checkpoints
rsync -avz lambda-overfit:~/Robust-Overfitting/checkpoints/ ./checkpoints/

# Download evaluation results
# report/evaluation_results.csv is generated by running scripts/evaluate.py
# (a separate step from training), which loads each saved checkpoint and scores
# it on the CIFAR-10 test set under both clean and PGD-20 adversarial evaluation.
# Columns: epoch, clean_loss, clean_acc, robust_loss, robust_acc, eval_time_sec
rsync -avz lambda-overfit:~/Robust-Overfitting/report/ ./report/

# Download TensorBoard logs
rsync -avz lambda-overfit:~/Robust-Overfitting/runs/ ./runs/
```

---

## 10. Terminating Instance to Stop Billing

> [!CAUTION]
> Lambda Labs instances are billed continuously while they are in **Booting** or **Running** state. Shutting down OS from within terminal (e.g., `sudo shutdown`) **does not** stop billing.

To stop incurring charges, follow instructions below:
1. Go to Lambda Labs dashboard under **Instances** tab.
2. Select checkbox next to your running instance.
3. Click **Terminate** button in top-right corner of dashboard.

> [!WARNING]
> Terminating an instance deletes all files and data stored on its local storage. Be sure to download any training logs, model checkpoints, or results to your local machine (using `rsync`) **before** terminating.
