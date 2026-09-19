import json
import re
import time
import urllib.parse
import urllib.request

from ..reliability import IntegrityError
from .pilot_runner import PilotProviderError, RunPodPilotProvider


class QualificationProvider(RunPodPilotProvider):
    def _rest(self, method, path, timeout):
        request = urllib.request.Request(
            "https://rest.runpod.io/v1/" + path, method=method,
            headers={"Authorization": "Bearer " + self.api_key})
        return self._json(request, timeout)

    def list_pods(self, timeout):
        result = self._rest("GET", "pods?includeMachine=true", timeout)
        if not isinstance(result, list):
            raise IntegrityError("qualification Pod inventory is not a list")
        return result

    def get_volume(self, volume_id, timeout):
        if volume_id != "xgogcmey5o":
            raise IntegrityError("qualification volume is outside allowed scope")
        return self._rest("GET", "networkvolumes/" + volume_id, timeout)

    def get_pod(self, pod_id, timeout):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", pod_id or ""):
            raise IntegrityError("qualification Pod ID is invalid")
        return self._rest("GET", "pods/" + pod_id + "?includeMachine=true", timeout)

    def create(self, payload, timeout):
        deadline = time.monotonic() + timeout
        query = "mutation($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id } }"
        url = "https://api.runpod.io/graphql?api_key=" + urllib.parse.quote(self.api_key, safe="")
        request = urllib.request.Request(url,
            data=json.dumps({"query": query, "variables": {"input": payload}}).encode(),
            headers={"Content-Type": "application/json"})
        response = self._json(request, timeout)
        if response.get("errors"):
            raise IntegrityError("qualification create GraphQL errors; reconcile without retry")
        pod_id = response.get("data", {}).get("podFindAndDeployOnDemand", {}).get("id")
        return self.get_pod(pod_id, max(0.001, deadline-time.monotonic()))

    def terminate(self, pod_id, timeout):
        if pod_id == self.pod_id or not re.fullmatch(r"[a-zA-Z0-9_-]+", pod_id or ""):
            raise IntegrityError("qualification cannot terminate protected or invalid Pod")
        return self._rest("DELETE", "pods/" + pod_id, timeout)

    def wait_absent(self, pod_id, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.get_pod(pod_id, min(10, deadline-time.monotonic()))
            except PilotProviderError as exc:
                if exc.safe_details.get("http_status") == 404:
                    return
                raise
            time.sleep(min(1, max(0, deadline-time.monotonic())))
        raise TimeoutError("disposable Pod absence was not externally confirmed")
