import argparse
import json
import time

import requests

V2_BASE_URL = "https://api.runpod.io/v2"
REQUEST_TIMEOUT = 30
RETRYABLE_STATUS_CODES = {429, 500, 501, 502, 503, 504}

TARGET_GPUS = [
    "NVIDIA RTX PRO 4500 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX PRO 4000 Blackwell",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA L40",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
]
DATA_CENTER_BAN_LIST = ["EUR-IS-1", "EUR-IS-2", "US-IL-1", "EU-CZ-1", "EUR-NO-1"]
CU130_IMAGES_NAME = "ghcr.io/xuehaosun/azure-agent:13.0"
MIN_CUDA_VERSION = "13.0"


def _response_payload(response):
    try:
        return response.json()
    except ValueError:
        return response.text


class RunPodAPIError(RuntimeError):
    def __init__(self, response):
        self.status_code = response.status_code
        self.payload = _response_payload(response)
        if isinstance(self.payload, dict):
            detail = self.payload.get("detail") or self.payload.get("title") or self.payload.get("error")
            errors = self.payload.get("errors")
            if errors:
                detail = f"{detail}: {json.dumps(errors)}"
        else:
            detail = self.payload
        super().__init__(f"RunPod API returned HTTP {self.status_code}: {detail}")


CAPACITY_ERROR_MARKERS = (
    "no longer any instances available",
    "no available instances",
    "no instances available",
    "insufficient capacity",
    "no capacity",
    "out of capacity",
)


def _is_capacity_error(error):
    if error.status_code != 400:
        return False
    payload = error.payload if isinstance(error.payload, str) else json.dumps(error.payload, default=str)
    payload = payload.lower()
    return any(marker in payload for marker in CAPACITY_ERROR_MARKERS)


class RunPodV2Client:
    def __init__(self, api_key, session=None):
        self.session = session if session is not None else requests.Session()
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def request(self, method, path, retries=3, retry_delay=5, **kwargs):
        headers = {**self.headers, **kwargs.pop("headers", {})}
        timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)

        for attempt in range(retries + 1):
            try:
                response = self.session.request(
                    method,
                    f"{V2_BASE_URL}{path}",
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                )
            except requests.exceptions.RequestException:
                if attempt >= retries:
                    raise
                print(f"⚠️ Request failed, retrying in {retry_delay} seconds ({attempt + 1}/{retries})...")
                time.sleep(retry_delay)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < retries:
                print(
                    f"⚠️ {response.status_code} Error, Retrying in {retry_delay} seconds "
                    f"({attempt + 1}/{retries})..."
                )
                time.sleep(retry_delay)
                continue

            if response.status_code >= 400:
                raise RunPodAPIError(response)
            return response

        raise RuntimeError(f"RunPod request failed after {retries} retries")

    def request_json(self, method, path, **kwargs):
        response = self.request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError:
            return {}


def run_create_pod(api_key, payload, client=None):
    client = client if client is not None else RunPodV2Client(api_key)
    return client.request_json("POST", "/pods", retries=3, retry_delay=90, json=payload)  # rp-migrate: ignore


def _parse_env(values):
    env = {}
    for value in values or []:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid environment variable '{value}', expected KEY=VALUE")
        env[key] = item
    return env


def _get_gpu_catalog(client, gpu_count, min_cuda_version):
    params = [
        ("include", "AVAILABILITY"),
        ("product", "POD"),
        ("count", str(gpu_count)),
        ("cloud", "SECURE"),
    ]
    params.append(("minCudaVersion", min_cuda_version))  # rp-migrate: ignore
    data = client.request_json("GET", "/catalog/gpus", params=params)
    gpus = data.get("gpus") if isinstance(data, dict) else None
    if not isinstance(gpus, list):
        raise RuntimeError("RunPod v2 GPU catalog response did not contain a 'gpus' list")
    return gpus


def _available_data_centers(gpu):
    return [
        data_center["id"]
        for data_center in gpu.get("dataCenters", [])
        if data_center.get("id")
        and data_center["id"] not in DATA_CENTER_BAN_LIST
        and data_center.get("availability", "NONE") != "NONE"
    ]


def _gpu_candidates(gpus):
    catalog_by_id = {gpu.get("id"): gpu for gpu in gpus if gpu.get("id")}
    candidates = []
    for gpu_id in TARGET_GPUS:
        gpu = catalog_by_id.get(gpu_id)
        if not gpu:
            continue
        data_center_ids = _available_data_centers(gpu)
        if data_center_ids:
            candidates.append((gpu_id, data_center_ids))
    return candidates


def _pod_payload(args, gpu_id, data_center_ids, env, min_cuda_version):
    return {
        "name": args.name,
        "image": CU130_IMAGES_NAME,
        "cloud": "SECURE",
        "disk": args.container_disk_size,
        "dataCenterIds": data_center_ids,
        "env": env,
        "gpu": {
            "id": gpu_id,
            "count": args.gpu_count,
            "minCudaVersion": min_cuda_version,  # rp-migrate: ignore
        },
    }


def create_pod(args):
    if not args.name:
        raise ValueError("--name is required for pod creation")
    if args.gpu_count < 1:
        raise ValueError("--gpu_count must be at least 1")

    env = _parse_env(args.env)
    min_cuda_version = MIN_CUDA_VERSION
    client = RunPodV2Client(args.api_key)
    gpus = _get_gpu_catalog(client, args.gpu_count, min_cuda_version)
    candidates = _gpu_candidates(gpus)
    if not candidates:
        raise RuntimeError("No requested GPU has availability in the selected data centers")

    print(f"🚀 Creating pod: {args.name}...")
    last_error = None
    for gpu_id, data_center_ids in candidates:
        payload = _pod_payload(args, gpu_id, data_center_ids, env, min_cuda_version)
        print(f"    Trying GPU: {gpu_id} in {', '.join(data_center_ids)}")
        try:
            data = run_create_pod(args.api_key, payload, client=client)
        except RunPodAPIError as error:
            if not _is_capacity_error(error):
                raise
            last_error = error
            print(f"⚠️ No capacity for {gpu_id}; trying the next available GPU. {error}")
            continue

        pod_id = data.get("id") if isinstance(data, dict) else None
        if not pod_id:
            raise RuntimeError("RunPod v2 create response did not contain a pod id")
        print(f"✅ Pod created successfully! Pod ID: {pod_id}")
        print(f"    Status is: {data.get('status', 'unknown')}")
        return

    if last_error:
        raise RuntimeError(f"No capacity for the requested GPUs. Last API error: {last_error}") from last_error
    raise RuntimeError("Failed to create pod")


def _list_pods(client):
    data = client.request_json("GET", "/pods")  # rp-migrate: ignore
    pods = data.get("pods") if isinstance(data, dict) else None  # rp-migrate: ignore
    if not isinstance(pods, list):
        raise RuntimeError("RunPod v2 pod list response did not contain a 'pods' list")  # rp-migrate: ignore
    return pods


def _get_pod_by_name(args, client):
    matches = [pod for pod in _list_pods(client) if pod.get("name") == args.name]
    return matches[0] if matches else None


def _get_pod_by_id(pod_id, client):
    try:
        return client.request_json("GET", f"/pods/{pod_id}")
    except RunPodAPIError as error:
        if error.status_code == 404:
            return None
        raise


def get_pod_id(args, client=None):
    client = client if client is not None else RunPodV2Client(args.api_key)
    pod = _get_pod_by_name(args, client)
    if pod:
        print(f"Pod status: {pod.get('status', 'unknown')}")
        return pod.get("id")
    print(f"⚠️ Pod '{args.name}' not found.")
    return None


def wait_for_pod(args):
    if not args.name:
        raise ValueError("--name is required while waiting for a pod")
    client = RunPodV2Client(args.api_key)
    for _ in range(60):
        pod = _get_pod_by_name(args, client)
        if pod:
            status = pod.get("status", "unknown")
            if status in {"ERROR", "EXITED", "TERMINATED"}:
                raise RuntimeError(f"Pod '{args.name}' entered terminal status {status}")
            if status == "RUNNING":
                print(f"✅ Pod '{args.name}' is now available with ID: {pod.get('id')} (status: {status})")
                return
            print(f"⏳ Pod '{args.name}' is {status}; waiting until it is RUNNING...")
        else:
            print(f"⏳ Waiting for pod '{args.name}' to be created...")
        time.sleep(10)
    print(f"❌ Timeout: Pod '{args.name}' was not created within the expected time.")
    raise RuntimeError(f"Pod '{args.name}' was not created within the expected time")


def terminate_pod(args):
    client = RunPodV2Client(args.api_key)
    pod_id = args.pod_id or get_pod_id(args, client)
    if not pod_id:
        raise RuntimeError(f"Pod '{args.name or 'unknown'}' not found")

    try:
        client.request("DELETE", f"/pods/{pod_id}", retries=30, retry_delay=5)
    except RunPodAPIError as error:
        if error.status_code != 404:
            raise
        print(f"✅ Pod {args.name or pod_id} was already terminated.")
        return

    for attempt in range(30):
        pod = _get_pod_by_id(pod_id, client)
        if pod is None:
            print(f"✅ Pod {args.name or pod_id} termination command sent.")
            return
        print(f"⚠️ Pod {args.name or pod_id}: {pod_id} termination initiated, but pod still exists")
        if attempt < 29:
            time.sleep(10)
    raise RuntimeError(f"Pod {args.name or pod_id}: termination may not have completed yet")


def main():
    parser = argparse.ArgumentParser(description="RunPod Pod Manager via API")
    parser.add_argument("--action", choices=["create", "terminate", "wait"], required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--pod_id", help="Pod ID for termination")
    parser.add_argument("--name", help="Pod name")
    parser.add_argument("--gpu_count", type=int, default=1)
    parser.add_argument("--container_disk_size", type=int, default=50)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--env", nargs="*", help="Environment variables in KEY=VALUE format")

    args = parser.parse_args()

    if args.action == "create":
        time.sleep(args.part * 15)
        create_pod(args)
    elif args.action == "terminate":
        terminate_pod(args)
    elif args.action == "wait":
        wait_for_pod(args)


if __name__ == "__main__":
    main()
