import threading
import queue
import time
from typing import Optional, Dict, Callable
import logging
from synco.main import run_pipeline, build_pipeline_config


logger = logging.getLogger(__name__)


class StreamlitPipelineExecutor:
    """
    Non-blocking wrapper around run_pipeline() using threading.
    Allows Streamlit UI to remain responsive during long-running pipeline execution.
    """

    def __init__(self, config: dict, on_step_change: Optional[Callable] = None):
        """
        Initialize the executor.

        Args:
            config: Configuration dictionary for the pipeline
            on_step_change: Optional callback when current step changes
        """
        self.config = config
        self.on_step_change = on_step_change
        self.thread: Optional[threading.Thread] = None
        self.result_queue: queue.Queue = queue.Queue()
        self.log_queue: queue.Queue = queue.Queue()
        self.status = "idle"  # idle, running, completed, error
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.current_step = 0
        self.artifacts = None
        self.error_message = None

    def start(self):
        """Start pipeline execution in a background thread."""
        if self.status == "running":
            logger.warning("Pipeline is already running")
            return

        self.status = "running"
        self.start_time = time.time()
        self.current_step = 0
        self.error_message = None
        self.artifacts = None

        # Clear queues
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break

        # Start background thread
        self.thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.thread.start()

    def _run_pipeline(self):
        """Execute the pipeline in a background thread."""
        try:
            self.log_queue.put({"type": "message", "text": "Building pipeline configuration..."})
            self.current_step = 1

            # Build final config
            merged_config = build_pipeline_config(self.config)
            self.log_queue.put(
                {"type": "message", "text": f"Configuration built successfully"}
            )

            # Run the pipeline
            self.log_queue.put({"type": "message", "text": "Starting pipeline execution..."})
            artifacts = run_pipeline(merged_config, verbose=True)

            self.current_step = 6
            self.artifacts = artifacts
            self.status = "completed"
            self.end_time = time.time()

            self.result_queue.put({
                "status": "completed",
                "artifacts": artifacts,
                "error": None,
            })

            self.log_queue.put(
                {"type": "success", "text": "Pipeline completed successfully"}
            )

        except Exception as e:
            logger.exception("Pipeline execution failed")
            self.status = "error"
            self.end_time = time.time()
            self.error_message = str(e)

            self.result_queue.put({
                "status": "error",
                "artifacts": None,
                "error": str(e),
            })

            self.log_queue.put({
                "type": "error",
                "text": f"Error: {str(e)}"
            })

    def get_result(self) -> Optional[Dict]:
        """
        Get the pipeline result if completed.

        Returns:
            Result dict with status, artifacts, and error (if any), or None if not ready
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def get_log_messages(self, max_messages: Optional[int] = None) -> list:
        """
        Get all available log messages from the queue.

        Args:
            max_messages: Optional limit on number of messages to retrieve

        Returns:
            List of log message dicts
        """
        messages = []
        count = 0

        while True:
            if max_messages and count >= max_messages:
                break

            try:
                msg = self.log_queue.get_nowait()
                messages.append(msg)
                count += 1
            except queue.Empty:
                break

        return messages

    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == "running" and (self.thread is None or self.thread.is_alive())

    def is_completed(self) -> bool:
        """Check if pipeline has completed successfully."""
        return self.status == "completed"

    def is_error(self) -> bool:
        """Check if pipeline encountered an error."""
        return self.status == "error"

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since pipeline started."""
        if self.start_time is None:
            return 0

        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def get_status_message(self) -> str:
        """Get human-readable status message."""
        if self.status == "idle":
            return "Ready to run"
        elif self.status == "running":
            return f"Running (Step {self.current_step}/6)"
        elif self.status == "completed":
            elapsed = self.get_elapsed_time()
            return f"Completed in {elapsed:.1f} seconds"
        elif self.status == "error":
            return f"Error: {self.error_message}"
        return "Unknown status"

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for pipeline to complete.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            True if completed, False if timeout
        """
        if self.thread is None:
            return False

        self.thread.join(timeout=timeout)
        return not self.is_running()
