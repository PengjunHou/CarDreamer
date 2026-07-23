import base64
import json
import logging
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template

# Target frame rate served to the browser. The env may step much faster; the
# stream renders only the most recent frame at this rate and drops the rest, so
# the web monitor never back-pressures or starves the training/eval loop.
_STREAM_FPS = 15


class EnvMonitorBase:
    def __init__(self, config):
        self._config = config
        # Single "latest frame" slot instead of unbounded queues: render() only
        # ever stores the most recent (obs, info) pair (older frames are dropped),
        # so the producer is O(1) and non-blocking and memory never grows.
        self._latest = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        # daemon=True: the Flask server ends with the interpreter. We deliberately
        # never join it (app.run never returns), so shutdown/crash cannot hang.
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self):
        # Silence the per-request access log (otherwise the SSE stream spams the
        # console/log with a GET line for every reconnect/poll).
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app = Flask(__name__, template_folder="templates")

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/stream")
        def stream():
            def generate():
                while True:
                    # Wake on a new frame, but time out periodically so a slow or
                    # idle producer keeps the SSE connection alive.
                    self._new_frame.wait(timeout=1.0)
                    self._new_frame.clear()
                    with self._lock:
                        latest = self._latest
                    if latest is None:
                        continue
                    obs, info = latest
                    frame = self._render(obs, info)
                    yield f"data: {json.dumps(frame)}\n\n"
                    # Cap the outgoing rate; intermediate frames are dropped.
                    time.sleep(1.0 / _STREAM_FPS)

            return Response(generate(), mimetype="text/event-stream")

        # threaded=True: a slow client or a second tab must not serialize behind
        # the long-lived /stream request.
        app.run(
            host="0.0.0.0",
            port=self._config.world.carla_port + 7000,
            use_reloader=False,
            threaded=True,
        )

    def _render_info(self, info):
        rendered_info = {}
        for key, value in info.items():
            if isinstance(value, (float, int, bool)):
                rendered_info[key] = value
            elif isinstance(value, np.number):
                rendered_info[key] = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                rendered_info[key] = value.tolist()
            else:
                rendered_info[key] = str(value)
        return rendered_info

    def _render_images(self, obs):
        images = []
        display_config = self._config.display
        if display_config.enable and display_config.render_keys:
            for key in display_config.render_keys:
                if key in obs:
                    img = obs[key]
                    if len(img.shape) == 2:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                    else:
                        img = img[:, :, ::-1]
                    _, img_encoded = cv2.imencode(".webp", img)
                    img_base64 = base64.b64encode(img_encoded).decode("utf-8")
                    images.append({"key": key, "image": img_base64})
        return images

    def _render(self, obs, info):
        return {"images": self._render_images(obs), "info": self._render_info(info)}

    def stop(self):
        # No-op: the server thread is a daemon and ends with the interpreter.
        # Never join it -- app.run() never returns, so joining would hang exit
        # (which is what turned crashed eval processes into port/GPU-holding
        # zombies).
        pass


class EnvMonitorOpenCV(EnvMonitorBase):
    def render(self, obs, info):
        # Store the latest frame and wake the stream. O(1), non-blocking; never
        # touches CARLA or JAX, so the env loop is never delayed by the monitor.
        with self._lock:
            self._latest = (obs, info)
        self._new_frame.set()
