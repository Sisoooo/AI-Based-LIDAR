# Inferenza pi0.5 nav su ws2 + test col robot simulato

Procedura **verificata** (ws2 = `130.251.13.151`, user `ter-ws-2`, host `k8s-worker-node-1`, 2026-06-25).
Pipeline: trainato su HPC (H200) → checkpoint → servito su GPU del lab → nodo ROS2 chiude il loop.

## File in questa cartella
- `inference_ros2_node.py` — nodo ROS2: cattura RViz + odom + prev_cmd → server → pubblica cmd_vel
- `mir_nav_policy.py` — transform openpi (1 cam RViz→base_0_rgb, state 9, action cmd_vel 3) → va in `openpi/src/openpi/policies/`
- `config_append_nav.py` — blocco da APPENDERE a `openpi/src/openpi/training/config.py` (registra `pi05_mir_nav`, auto-register)
- `serve_nav.sh` — avvia il policy server
- `test_client.py` — test rapido del server (senza robot)
- `config_mir_nav.py`, `nav_policy.py` — sorgenti originali (riferimento)

## Setup openpi su ws2 (una volta)
```bash
cd ~/vla_nav
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi && uv sync --no-install-package rerun-sdk
# policy + config (i file di questa cartella, via base64 o scp):
cp /path/mir_nav_policy.py src/openpi/policies/mir_nav_policy.py
cat /path/config_append_nav.py >> src/openpi/training/config.py
UV_NO_SYNC=1 uv run --no-sync python -c "from openpi.training import config as c; c.get_config('pi05_mir_nav'); print('config OK')"
```
Checkpoint (solo params+assets dello step finale, NON train_state) in `~/vla_nav/ckpt/mir_nav/29999/` (params/ + assets/).

## 1) Avvia il server (su ws2, in tmux per lasciarlo su)
```bash
tmux new -s serve
cd ~/vla_nav/openpi && bash /path/serve_nav.sh        # -> "server listening on 0.0.0.0:8000"
# stacca: Ctrl-b d   |   riattacca: tmux attach -t serve
```
⚠️ `XLA_PYTHON_CLIENT_PREALLOCATE=false` (già nello script) è essenziale: la GPU è condivisa con vLLM/Qwen.

## 2) Test rapido (senza robot, altra shell su ws2)
```bash
cd ~/vla_nav/openpi && UV_NO_SYNC=1 uv run --no-sync python /path/test_client.py
# atteso: OK - action shape (10, 3) | primo step: [vx, ~0, wz]
```

## 3) Test col robot simulato (nodo ROS2)
Sulla macchina dove gira **RViz** (il nodo cattura lo schermo) + il sim MiR che pubblica `/diff_cont/odom`:
```bash
source /opt/ros/<distro>/setup.bash
# serve: pip install openpi-client opencv-python mss  (nell'env python di ROS)
python3 inference_ros2_node.py --ros-args \
  -p policy_host:=127.0.0.1 \           # 130.251.13.151 se il nodo NON è su ws2
  -p prompt:="reach the red square" \
  -p actions_per_query:=1               # receding horizon (reattivo)
```
Cambia `-p prompt:=` per dare obiettivi diversi.

## Input del modello (cosa manda il nodo)
- `observation/image` — frame RViz `(1200,1920,3)` uint8 RGB
- `observation/state` — `(9,)` float32: `[odom_x, odom_y, yaw, vx, vy, wz, prev_cmd_x, prev_cmd_y, prev_cmd_z]`
  (i 3 `prev_cmd_*` = l'ULTIMO cmd_vel pubblicato; il nodo li traccia, parte da 0)
- `prompt` — `"reach the {color} {shape}"` · color ∈ {blue,green,red,yellow} · shape ∈ {dot,triangle,square}

Output: `action (10, 3)` = 10 passi di `cmd_vel` `[linear_x, linear_y(~0 diff-drive), angular_z]`. Il nodo pubblica `action[0]` su `/diff_cont/cmd_vel_unstamped` e re-interroga.

## Gotcha
- Al **serving** openpi NON applica il repack → il client manda le chiavi INTERNE (`observation/image`, `observation/state`, `prompt`), NON quelle lerobot (`observation.images.rviz`...).
- Una GPU 48 GB basta (~6 GB bf16). Se OOM: GPU occupata → `nvidia-smi`, libera o `PREALLOCATE=false` (già attivo).
- Il modello vede ciò che vede RViz: per funzionare, RViz deve mostrare la scena come durante la raccolta dati (stessa regione/risoluzione).
