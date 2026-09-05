import json
import os
import re
import time
import urllib.error
import urllib.request


class RunPodShutdownError(RuntimeError):
    pass


def stop_current_pod(*, timeout=15, attempts=3, sleep=time.sleep):
    pod_id = os.getenv("RUNPOD_POD_ID")
    if not pod_id:
        return {"requested": False, "reason": "not_running_on_runpod"}
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise RunPodShutdownError("RUNPOD_POD_ID contains invalid characters")
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise RunPodShutdownError("RUNPOD_API_KEY is required to stop the active pod")
    if "\r" in api_key or "\n" in api_key:
        raise RunPodShutdownError("RUNPOD_API_KEY must not contain line breaks")
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if response.status != 200:
                    raise RunPodShutdownError(f"RunPod stop returned HTTP {response.status}")
                return {"requested": True, "pod_id": pod_id,
                        "response": json.loads(payload) if payload else None}
        except (urllib.error.URLError, TimeoutError, RunPodShutdownError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                sleep(2 ** attempt)
    raise RunPodShutdownError(f"RunPod stop failed after {attempts} attempts: {last_error}")
