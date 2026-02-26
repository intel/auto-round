import requests
import argparse
import sys
import json
import time


TARGET_GPUS = [
    # "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA GeForce RTX 5090"
]
REQUIRED_COUNT = 1


def check_gpu_count(token):
    url = f"https://api.runpod.io/graphql?api_key={token}"
    ids_string = ", ".join([f'"{gid}"' for gid in TARGET_GPUS])
    graphql_query = """
    query GpuAvailability($input: GpuLowestPriceInput!) {
      gpuTypes(input: {ids: [%s]}) {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        maxGpuCount
        maxGpuCountSecureCloud
        maxGpuCountCommunityCloud
        minPodGpuCount
        lowestPrice(input: $input) {
          gpuName
          stockStatus
          minimumBidPrice
          uninterruptablePrice
          maxGpuCount
          maxUnreservedGpuCount
          availableGpuCounts
          rentedCount
          totalCount
        }
      }
    }
    """ % ids_string

    variables = {"input": {"gpuCount": 1, "secureCloud": True, "minMemoryInGb": 0, "minVcpuCount": 0}}

    try:
        response = requests.post(
            url, json={"query": graphql_query, "variables": variables}, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        all_gpus = data.get("data", {}).get("gpuTypes", [])

        id_to_gpu = {gpu['id']: gpu for gpu in all_gpus}
        all_gpus = [id_to_gpu[gpu_id] for gpu_id in TARGET_GPUS if gpu_id in id_to_gpu]

        print(f"--- Checking target graphics card inventory ---\n")

        for gpu in all_gpus:
            gpu_id = gpu.get("id")

            if gpu_id in TARGET_GPUS:
                max_count = gpu.get("lowestPrice", {}).get("maxUnreservedGpuCount", 0)

                if REQUIRED_COUNT > max_count:
                    print(
                        f"❌ {gpu_id}: \n   Status: Insufficient inventory.\n"
                    )
                    continue
                else:
                    print(f"✅ {gpu_id}: \n   Status: Sufficient inventory.\n")
                    return gpu_id

        print("❌ No compliant GPU found.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"HTTP Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_create_pod(api_key, payload):
    url = "https://rest.runpod.io/v1/pods"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    response.raise_for_status()
    if response.status_code != 201:
        print(f"❌ HTTP Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()
    if "errors" in result:
        print("❌ Errors:")
        print(json.dumps(result["errors"], indent=2))
        sys.exit(1)
    return result


def create_pod(args):
    if args.env:
        env_dict = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in args.env}

    gpu_type = None
    for _ in range(10):  # Try up to 10 times to find an available GPU
        gpu_type = check_gpu_count(args.api_key)
        if gpu_type:
            break
        else:
            print("⏳ No compliant GPU available. Retrying in 5 minutes...")
            time.sleep(60 * 5)

    if not gpu_type:
        print("❌ No compliant GPU found after multiple attempts. Exiting.")
        sys.exit(1)

    payload = {
        "cloudType": "SECURE",
        "containerDiskInGb": args.container_disk_size,
        "env": env_dict,
        "gpuCount": args.gpu_count,
        "gpuTypeIds": [gpu_type],
        "name": args.name,
        "volumeInGb": 0,
        "templateId": "qtac2hjxru"
    }

    print(f"🚀 Creating pod: {args.name}...")
    data = run_create_pod(args.api_key, payload)
    if data:
        pod_id = data.get("id")
        if pod_id:
            print(f"✅ Pod created successfully! Pod ID: {pod_id}")
            print(f"    Status is: {data.get('desiredStatus')}")
    else:
        print("❌ Failed to create pod (no data returned).")
        sys.exit(1)


def get_pod_id(args):
    url = f"https://rest.runpod.io/v1/pods?name={args.name}"
    headers = {"Authorization": "Bearer " + args.api_key}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()[0] if response.json() else None
        if data:
            print(f"Pod status: {data.get('desiredStatus', 'unknown')}")
        if data and "id" in data:
            return data["id"]

        print(f"⚠️ Pod '{args.name}' not found.")
        return None

    except Exception as e:
        print(f"⚠️ Error fetching pods: {e}")
        raise e


def wait_for_pod(args):
    for _ in range(60):  # Wait up to 10 minutes
        pod_id = get_pod_id(args)
        if pod_id:
            print(f"✅ Pod '{args.name}' is now available with ID: {pod_id}")
            return
        else:
            print(f"⏳ Waiting for pod '{args.name}' to be created...")
            time.sleep(10)
    print(f"❌ Timeout: Pod '{args.name}' was not created within the expected time.")
    sys.exit(1)


def terminate_pod(args):
    pod_id = args.pod_id or get_pod_id(args)
    if not pod_id:
        get_pod_id(args)  # Just to check if pod exists and print status
        sys.exit(1)

    url = f"https://rest.runpod.io/v1/pods/{pod_id}"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    response = requests.delete(url, headers=headers)
    response.raise_for_status()

    max_tries = 30

    for i in range(max_tries):  # Wait up to 5 minutes for termination
        pod_id = get_pod_id(args)
        if pod_id:
            print(f"⚠️ Pod {args.name}: {pod_id} termination initiated, but pod still exists")
            if i >= max_tries - 1:
                raise Exception(
                    f"❌ Pod {args.name}: {pod_id} termination may not have completed yet. Please check the status."
                )
        else:
            print(f"✅ Pod {args.name} termination command sent.")
            break
        time.sleep(10)  # Wait a bit for termination to process


def main():
    parser = argparse.ArgumentParser(description="RunPod Pod Manager via API")
    parser.add_argument("--action", choices=["create", "terminate", "wait"], required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--pod_id", help="Pod ID for termination")
    parser.add_argument("--name", help="Pod name")
    parser.add_argument("--gpu_count", type=int, default=1)
    parser.add_argument("--container_disk_size", type=int, default=50)
    parser.add_argument("--env", nargs="*", help="Environment variables in KEY=VALUE format")

    args = parser.parse_args()

    if args.action == "create":
        create_pod(args)
    elif args.action == "terminate":
        terminate_pod(args)
    elif args.action == "wait":
        wait_for_pod(args)


if __name__ == "__main__":
    main()
