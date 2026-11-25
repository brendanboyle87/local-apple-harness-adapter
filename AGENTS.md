# Repository instructions

## Apple Silicon memory preflight

Before loading a model for training, evaluation, or profiling, run `memory_pressure -Q` on the Mac and check for active MLX trainers and local inference servers. Check loaded-model status through the server when available. An idle server process can still hold model weights in unified memory.

Run one model workload at a time. Compare current memory pressure and available headroom with the most relevant measured MLX peak, allowing room for macOS, Docker, and other active work. Unload idle models or stop stale workers only when they belong to this run or the user has authorized it. Recheck memory pressure and loaded-model status before starting. If unrelated work is consuming the needed headroom, report the specific blocker.

Before resuming, verify the previous trainer has exited and identify the latest committed group from its metrics and training state. When the run finishes or the user asks to stop, verify that the trainer has exited and any models loaded for this work have been released.
