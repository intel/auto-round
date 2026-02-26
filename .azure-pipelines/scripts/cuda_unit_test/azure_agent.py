import time
import requests
import argparse
import sys
import base64


API_VERSION = "7.1"


def get_auth_header(pat):
    return {"Authorization": "Basic " + base64.b64encode(f":{pat}".encode()).decode()}


def get_pool_id(organization_url, pat, pool_name):
    url = f"{organization_url}/_apis/distributedtask/pools?poolName={pool_name}&api-version={API_VERSION}"
    headers = get_auth_header(pat)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if data["count"] > 0:
        return data["value"][0]["id"]
    return None


def wait_for_agent(organization_url, pat, pool_name, agent_name, timeout_seconds=1200):
    start_time = time.time()
    sleep_interval = 5

    print(f"Waiting for agent '{agent_name}' to come online in pool '{pool_name}'...")

    pool_id = get_pool_id(organization_url, pat, pool_name)
    if not pool_id:
        print(f"Error: Pool '{pool_name}' not found.")
        sys.exit(1)

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"❌ Timeout: Agent failed to come online within {timeout_seconds} seconds.")
            sys.exit(1)

        url = f"{organization_url}/_apis/distributedtask/pools/{pool_id}/agents?agentName={agent_name}&api-version={API_VERSION}"
        headers = get_auth_header(pat)
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data["count"] > 0:
                agent = data["value"][0]
                status = agent.get("status")
                if status == "online":
                    print(f"✅ Agent: {agent_name} is ONLINE and ready!")
                    return
                else:
                    print(f"⏳ Agent status: '{status}'. Waiting...")
            else:
                print(f"⏳ Agent '{agent_name}' not found yet. Waiting...")

        except Exception as e:
            print(f"⚠️ Error checking agent status: {e}")

        time.sleep(sleep_interval)


def deregister_agent(organization_url, pat, pool_name, agent_name):
    print(f"Attempting to deregister agent '{agent_name}' from pool '{pool_name}'...")

    pool_id = get_pool_id(organization_url, pat, pool_name)
    if not pool_id:
        print(f"Pool '{pool_name}' not found. Cannot deregister agent.")
        return

    url = f"{organization_url}/_apis/distributedtask/pools/{pool_id}/agents?agentName={agent_name}&api-version={API_VERSION}"
    headers = get_auth_header(pat)

    try:
        for i in range(5):
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to query agent. Status: {response.status_code}")
                return

            data = response.json()
            if data["count"] == 0:
                if i < 4:
                    print(f"⚠️ Agent: {agent_name} not found in the pool (It might have cleaned itself up). Retrying...")
                    time.sleep(5)
                    continue
                else:
                    print(f"✅ Agent: {agent_name} not found in the pool (It might have cleaned itself up). Exiting.")
                    return

        agent_id = data["value"][0]["id"]
        print(f"Found Agent ID: {agent_id}")

        delete_url = (
            f"{organization_url}/_apis/distributedtask/pools/{pool_id}/agents/{agent_id}?api-version={API_VERSION}"
        )
        delete_response = requests.delete(delete_url, headers=headers)

        if delete_response.status_code == 204 or delete_response.status_code == 200:
            print("✅ Agent successfully removed from Azure DevOps.")
        else:
            print(f"❌ Failed to delete agent. Status: {delete_response.status_code}, Response: {delete_response.text}")

    except Exception as e:
        print(f"Error during deregistration: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Azure DevOps Agents")
    parser.add_argument("--action", required=True, choices=["wait", "deregister"], help="Action to perform")
    parser.add_argument("--url", required=True, help="Azure DevOps Organization URL (System.CollectionUri)")
    parser.add_argument("--pat", required=True, help="Azure DevOps PAT")
    parser.add_argument("--pool", required=True, help="Agent Pool Name")
    parser.add_argument("--agent", required=True, help="Agent Name")

    args = parser.parse_args()

    if args.action == "wait":
        wait_for_agent(args.url, args.pat, args.pool, args.agent)
    elif args.action == "deregister":
        deregister_agent(args.url, args.pat, args.pool, args.agent)
    else:
        print(f"Unknown action: {args.action}")
        sys.exit(1)
