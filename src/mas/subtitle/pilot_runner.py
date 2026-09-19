import json
import time
import urllib.error
import urllib.parse
import urllib.request

from ..reliability import IntegrityError


class PilotProviderError(IntegrityError):
    def __init__(self, details):
        self.safe_details = details
        super().__init__("RunPod provider request failed: " + json.dumps(details, sort_keys=True))


class RunPodPilotProvider:
    def __init__(self, pod_id, api_key):
        self.pod_id = pod_id
        self.api_key = api_key

    def _json(self, request, timeout):
        request.add_header("User-Agent", "ma-sub/1.0")
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if not payload and request.get_method() in ("POST", "DELETE"):
                    return {}
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(4096).decode("utf-8", "replace").lower()
            except OSError:
                detail = ""
            category = "capacity" if "not enough free gpu" in detail else "http_error"
            raise PilotProviderError({
                "http_status": exc.code,
                "category": category,
                "method": request.get_method(),
            }) from None
        except (OSError, ValueError) as exc:
            raise PilotProviderError({
                "error_type": type(exc).__name__,
                "method": request.get_method(),
            }) from None

    def account(self, timeout=15):
        query = json.dumps({
            "query": "query { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"
        }).encode()
        url = "https://api.runpod.io/graphql?api_key=" + urllib.parse.quote(self.api_key, safe="")
        result = self._json(urllib.request.Request(
            url,
            data=query,
            headers={"Content-Type": "application/json"},
        ), timeout)
        if result.get("errors") or not isinstance(result.get("data", {}).get("myself"), dict):
            raise IntegrityError("RunPod billing query failed")
        return result["data"]["myself"]

    def _pod_request(self, method, suffix="", timeout=15):
        request = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{self.pod_id}{suffix}",
            method=method,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return self._json(request, timeout)

    def pod(self, timeout=15):
        return self._pod_request("GET", timeout=timeout)

    def start(self, timeout):
        return self._pod_request("POST", "/start", min(15, timeout))

    def stop(self, timeout=15):
        return self._pod_request("POST", "/stop", timeout)

    def _wait(self, status, timeout):
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"RunPod did not reach {status}")
            value = self.pod(min(15, left))
            if value.get("desiredStatus") == status:
                return value
            time.sleep(min(2, left))

    def wait_running(self, timeout):
        return self._wait("RUNNING", timeout)

    def wait_exited(self, timeout):
        return self._wait("EXITED", timeout)
