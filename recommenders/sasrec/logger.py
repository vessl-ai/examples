import vessl


class VesslLogger:
    """VESSL logger"""
    def __init__(self):
        """Initializer"""
        self._log = {}

    def log(self, step, metric, value):
        """Log metrics. Each metric's log will be stored in the corresponding list.
        Args:
            metric (str): Metric name.
            value (float): Value.
        """
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)
        vessl.log(step=step, payload={
            metric: value,
        })

    def get_log(self):
        """Getter
        Returns:
            dict: Log metrics.
        """
        return self._log
