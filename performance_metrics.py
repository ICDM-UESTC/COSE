import pytorch_lightning as pl
import torch

class PerformanceMetricsCallback(pl.Callback):
    """
    A PyTorch Lightning Callback to precisely measure step duration and peak memory usage.
    """
    def __init__(self):
        super().__init__()
        self.start_event = None
        self.end_event = None
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_event.record()
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()       
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
            
            # Log the metrics
            metrics = {
                'step_duration_ms': elapsed_time_ms,
                'peak_memory_mb': peak_memory_mb
            }
            trainer.logger.log_metrics(metrics, step=trainer.global_step)