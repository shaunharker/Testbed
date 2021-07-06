# genesiscloud.py
# BSD License
# Python wrapper for api.genesiscloud.com
# See https://developers.genesiscloud.com and their excellent documentation.

# Change Log
#
# Shaun Harker 2021-07-05
# I've read genesiscloud's documentation of their API and have wrapped it in
# python in a manner that seemed sensible to me. I've included some of
# their documentation in the docstrings, occasionally modified to suit
# the python context.

import json
import time
import os
from requests import get as GET, post as POST, put as PUT, patch as PATCH, delete as DELETE

class Client:
    def __init__(self):
        # Go to https://account.genesiscloud.com/dashboard/security
        # to obtain your secret API token. DO NOT COMMIT IT TO A REPOSITORY.
        self.API_TOKEN = os.getenv("GENESISCLOUD_APIKEY")
        self.headers = {"Content-Type": "application/json",
                        "X-Auth-Token": self.API_TOKEN}
        self.time_of_last_api_call = time.time()

    def instance_types(self):
        return instance_types()

    # Instances. See https://developers.genesiscloud.com/instances
    def create_instance(self, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#create-an-instance
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)

        return self.api(verb=POST,
                        expected_status_code=201,
                        endpoint="https://api.genesiscloud.com/compute/v1/instance",
                        body_parameters=body_parameters)

    def list_all_instances(self, query_parameters=None, **kwargs): #per_page=50, page=1):
        """
        https://developers.genesiscloud.com/instances#list-all-instances
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)

        try:
            for key in query_parameters:
                assert key in ["per_page", "page"]
        except AssertionError:
            raise ValueError(f"Invalid arguments: {kwargs}\n"
                              "Please see https://developers.genesiscloud.com/instances#list-all-instances")

        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint="https://api.genesiscloud.com/compute/v1/instances",
                        query_parameters=query_parameters)

    def get_instance(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#get-instance
        """
        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}")

    def destroy_an_instance(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#destroy-an-instance
        """
        # TODO: Consider an asynchronous version.
        # Refuse to destroy while instance is in "copying" state.
        while self.get_instance(instance_id)["status"] == "copying":
            time.sleep(1.0)
        return self.api(verb=DELETE,
                        expected_status_code=204,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}")

    def snapshot_an_instance(self, instance_id, body_parameters, **kwargs): #snapshot_name):
        """
        https://developers.genesiscloud.com/instances#snapshot-an-instance
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)
        try:
            for key in body_parameters:
                assert key in ["name"]
        except AssertionError:
            raise ValueError(f"Invalid body parameter {key}\n"
                              "Please see https://developers.genesiscloud.com/instances#snapshot-an-instance")
        return self.api(verb=POST,
                        expected_status_code=201,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/snapshots",
                        body_parameters=body_parameters)

    def list_snapshots_of_an_instance(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#list-snapshots-of-an-instance
        """

        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/snapshots")

    def attachdetach_security_groups_from_an_instance(self, instance_id, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#attachdetach-security-groups-from-an-instance
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)

        return self.api(verb=PATCH,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}",
                        body_parameters=body_parameters)

    def attachdetach_volumes_from_an_instance(self, instance_id, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#attachdetach-volumes-from-an-instance
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)

        return self.api(verb=PATCH,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}",
                        body_parameters=body_parameters)

    def update_an_instance(self, instance_id, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#update-an-instance
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)

        return self.api(verb=PATCH,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}",
                        body_parameters=body_parameters)

    def get_instance_actions(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#get-instance-actions
        """
        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/actions")

    def perform_action(self, instance_id, body_parameters=None, **kwarg):
        """
        https://developers.genesiscloud.com/instances#perform-action
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)
        action = body_parameters["action"]
        assert action in ["stop", "start", "reset"]

        return self.api(verb=POST,
                        expected_status_code=204,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/actions",
                        body_parameters=body_parameters)

    # Images. See https://developers.genesiscloud.com/images
    def list_images(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/images#list-images
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)

        try:
            for key in query_parameters:
                assert key in ["type", "per_page", "page"]
                if key == "type":
                    assert query_parameters["type"] in ["base-os", "preconfigured", "snapshot"]
        except AssertionError:
            raise ValueError(f"Invalid arguments: {kwargs}\n"
                              "Please see https://developers.genesiscloud.com/images#list-images")
        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/images",
                        query_parameters=query_parameters)

    # Snapshots. https://developers.genesiscloud.com/snapshots
    def list_snapshots(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/snapshots#list-snapshots
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)

        try:
            for key in query_parameters:
                assert key in ["per_page", "page"]
        except AssertionError:
            raise ValueError(f"Invalid arguments: {query_parameters}\n"
                              "Please see https://developers.genesiscloud.com/snapshots#list-snapshots")
        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/snapshots",
                        query_parameters=query_parameters)

    def get_snapshot(self, snapshot_id):
        """
        https://developers.genesiscloud.com/snapshots#get-snapshot
        """
        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/snapshots/{snapshot_id}")

    def delete_a_snapshot(self, snapshot_id):
        """
        https://developers.genesiscloud.com/snapshots#delete-a-snapshot
        """
        return self.api(verb=DELETE,
                        expected_status_code=204,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/snapshots/{snapshot_id}")

    # Volumes. See https://developers.genesiscloud.com/volumes
    def create_a_volume(self, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/volumes#create-a-volume
        """
        body_parameters = self.parse_args(query_parameters, **kwargs)

        return self.api(verb=POST,
                        expected_status_code=201,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/volumes",
                        body_parameters=body_parameters)

    def list_volumes(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/volumes#list-volumes
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)

        try:
            for key in query_parameters:
                assert key in ["per_page", "page"]
        except AssertionError:
            raise ValueError(f"Invalid arguments: {kwargs}\n"
                              "Please see https://developers.genesiscloud.com/volumes#list-volumes")

        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/volumes",
                        query_parameters=query_parameters)

    def get_volume(self, volume_id):
        """
        https://developers.genesiscloud.com/volumes#get-volume
        """
        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/volumes/{volume_id}")

    def delete_a_volume(self, volume_id):
        """
        https://developers.genesiscloud.com/volumes#delete-a-volume
        """
        return self.api(verb=DELETE,
                        expected_status_code=204,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/volumes/{volume_id}")


    # SSH Keys. See https://developers.genesiscloud.com/ssh-keys
    def list_ssh_keys(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/ssh-keys#list-ssh-keys
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        try:
            for key in query_parameters:
                assert key in ["per_page", "page"]
        except AssertionError:
            raise ValueError(f"Invalid arguments: {query_parameters}\n"
                              "Please see https://developers.genesiscloud.com/ssh-keys")

        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/ssh-keys",
                        query_parameters=query_parameters)

    # Security Groups. See https://developers.genesiscloud.com/security-groups
    def create_security_groups(self, body_parameters, **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#create-security-groups
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)
        return self.api(verb=POST,
                        expected_status_code=201,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups",
                        body_parameters=body_parameters)

    def update_security_groups(self, body_parameters, **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#update-security-groups
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)
        return self.api(verb=PUT,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups",
                        body_parameters=body_parameters)

    def list_security_groups(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#list-security-groups
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        try:
            for key in query_parameters:
                assert key in ["per_page", "page"]
        except AssertionError:
            raise ValueError(f"Invalid arguments: {query_parameters}\n"
                              "Please see https://developers.genesiscloud.com/security-groups#list-security-groups")

        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups",
                        query_parameters=query_parameters)

    def get_security_group(self, security_group_id):
        """
        https://developers.genesiscloud.com/security-groups#get-security-group
        """
        return self.api(verb=GET,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups/{security_group_id}")

    def delete_a_security_group(self, security_group_id):
        """
        https://developers.genesiscloud.com/security-groups#delete-a-security-group
        """
        return self.api(verb=DELETE,
                        expected_status_code=204,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups/{security_group_id}")

    def parse_args(self, parameters=None, **kwargs):
        if parameters is None:
            parameters = kwargs
        else:
            for key in kwargs:
                parameters[key] = kwargs[key]
        return parameters

    def api(self, verb, expected_status_code, endpoint, query_parameters=None, body_parameters=None):
        """
        Call used by the other methods to interact with api.genesiscloud.com
        """
        # This ensures we won't hit the API rate limit of 10Hz,
        # unless class users break the singleton pattern:
        #  (see https://developers.genesiscloud.com/#rate-limiting)
        before = self.time_of_last_api_call
        now = time.time()
        if now - before < 0:
            before = now
        if now < before + 0.1:  # rate limit
            time.sleep(0.1 - (now - before))
        self.time_of_last_api_call = now

        response = verb(
            f"https://api.genesiscloud.com/compute/v1/{endpoint}",
            headers=self.headers,
            params=query_parameters,
            json=body_parameters,
        )
        if response.status_code != expected_status_code:
            print(response.status_code)
            print(json.dumps(response.json(), indent=4, sort_keys=True))
            raise RuntimeError(f"Status code {response.status_code} from api.genesiscloud.com\n"
                               f"with response of {json.dumps(response.json(), indent=4, sort_keys=True)}")
        return response.json()

def instance_types():
    data [("Type (Product Name)", "API Instance Type Identifier", "vCPUs", "Memory", "Disk", "GPUs", "Price on-demand (per hour)"),
        ("GPU Instance 1x GeForce™ RTX 3090", "vcpu-4_memory-18g_disk-80g_nvidia3090-1", "4", "18 GiB", "80 GiB", "1"),
        ("GPU Instance 2x GeForce™ RTX 3090", "vcpu-8_memory-36g_disk-80g_nvidia3090-2", "8", "36 GiB", "80 GiB", "2"),
        ("GPU Instance 3x GeForce™ RTX 3090", "vcpu-12_memory-54g_disk-80g_nvidia3090-3", "12", "54 GiB", "80 GiB" ,"3"),
        ("GPU Instance 4x GeForce™ RTX 3090", "vcpu-16_memory-72g_disk-80g_nvidia3090-4", "16", "72 GiB", "80 GiB" ,"4"),
        ("GPU Instance 5x GeForce™ RTX 3090", "vcpu-20_memory-90g_disk-80g_nvidia3090-5", "20", "90 GiB", "80 GiB" ,"5"),
        ("GPU Instance 6x GeForce™ RTX 3090", "vcpu-24_memory-108g_disk-80g_nvidia3090-6", "24", "108 GiB", "80 GiB" ,"6"),
        ("GPU Instance 7x GeForce™ RTX 3090", "vcpu-28_memory-126g_disk-80g_nvidia3090-7", "28", "126 GiB", "80 GiB" ,"7"),
        ("GPU Instance 8x GeForce™ RTX 3090", "vcpu-32_memory-144g_disk-80g_nvidia3090-8", "32", "144 GiB", "80 GiB", "8"),
        ("GPU Instance 1x GeForce™ RTX 3080", "vcpu-4_memory-12g_disk-80g_nvidia3080-1", "4", "12 GiB", "80 GiB", "1"),
        ("GPU Instance 2x GeForce™ RTX 3080", "vcpu-8_memory-24g_disk-80g_nvidia3080-2", "8", "24 GiB", "80 GiB", "2"),
        ("GPU Instance 3x GeForce™ RTX 3080", "vcpu-12_memory-36g_disk-80g_nvidia3080-3", "12", "36 GiB", "80 GiB", "3"),
        ("GPU Instance 4x GeForce™ RTX 3080", "vcpu-16_memory-48g_disk-80g_nvidia3080-4", "16", "48 GiB", "80 GiB", "4"),
        ("GPU Instance 5x GeForce™ RTX 3080", "vcpu-20_memory-60g_disk-80g_nvidia3080-5", "20", "60 GiB", "80 GiB", "5"),
        ("GPU Instance 6x GeForce™ RTX 3080", "vcpu-24_memory-72g_disk-80g_nvidia3080-6", "24", "72 GiB", "80 GiB", "6"),
        ("GPU Instance 7x GeForce™ RTX 3080", "vcpu-28_memory-84g_disk-80g_nvidia3080-7", "28", "84 GiB", "80 GiB", "7"),
        ("GPU Instance 8x GeForce™ RTX 3080", "vcpu-32_memory-96g_disk-80g_nvidia3080-8", "32", "96 GiB", "80 GiB", "8"),
        ("GPU Instance 1x NVIDIA 1080Ti", "vcpu-4_memory-12g_disk-80g_nvidia1080ti-1", "4", "12 GiB", "80 GiB", "1"),
        ("GPU Instance 2x NVIDIA 1080Ti", "vcpu-8_memory-24g_disk-80g_nvidia1080ti-2", "8", "24 GiB", "80 GiB", "2"),
        ("GPU Instance 3x NVIDIA 1080Ti", "vcpu-12_memory-36g_disk-80g_nvidia1080ti-3", "12", "36 GiB", "80 GiB", "3"),
        ("GPU Instance 4x NVIDIA 1080Ti", "vcpu-16_memory-48g_disk-80g_nvidia1080ti-4", "16", "48 GiB", "80 GiB", "4"),
        ("GPU Instance 5x NVIDIA 1080Ti", "vcpu-20_memory-60g_disk-80g_nvidia1080ti-5", "20", "60 GiB", "80 GiB", "5"),
        ("GPU Instance 6x NVIDIA 1080Ti", "vcpu-24_memory-72g_disk-80g_nvidia1080ti-6", "24", "72 GiB", "80 GiB", "6"),
        ("GPU Instance 7x NVIDIA 1080Ti", "vcpu-28_memory-84g_disk-80g_nvidia1080ti-7", "28", "84 GiB", "80 GiB", "7"),
        ("GPU Instance 8x NVIDIA 1080Ti", "vcpu-32_memory-96g_disk-80g_nvidia1080ti-8", "32", "96 GiB", "80 GiB", "8"),
        ("GPU Instance 1x AMD MI25", "vcpu-4_memory-24g_disk-80g_amdmi25-1", "4", "24 GiB", "80 GiB", "1"),
        ("GPU Instance 2x AMD MI25", "vcpu-8_memory-48g_disk-80g_amdmi25-2", "8", "48 GiB", "80 GiB", "2"),
        ("GPU Instance 3x AMD MI25", "vcpu-14_memory-72g_disk-80g_amdmi25-3", "14", "72 GiB", "80 GiB", "3"),
        ("GPU Instance 4x AMD MI25", "vcpu-18_memory-96g_disk-80g_amdmi25-4", "18", "96 GiB", "80 GiB", "4"),
        ("GPU Instance 5x AMD MI25", "vcpu-24_memory-120g_disk-80g_amdmi25-5", "24", "120 GiB", "80 GiB", "5"),
        ("GPU Instance 6x AMD MI25", "vcpu-28_memory-144g_disk-80g_amdmi25-6", "28", "144 GiB", "80 GiB", "6"),
        ("GPU Instance 7x AMD MI25", "vcpu-32_memory-168g_disk-80g_amdmi25-7", "32", "168 GiB", "80 GiB", "7"),
        ("GPU Instance 8x AMD MI25", "vcpu-38_memory-192g_disk-80g_amdmi25-8", "38", "192 GiB", "80 GiB", "8"),
        ("GPU Instance 9x AMD MI25", "vcpu-42_memory-216g_disk-80g_amdmi25-9", "42", "216 GiB", "80 GiB", "9"),
        ("GPU Instance 10x AMD MI25", "vcpu-48_memory-240g_disk-80g_amdmi25-10", "48", "240 GiB", "80 GiB", "10"),
        ("GPU Instance 4x AMD RX470", "vcpu-2_memory-8g_disk-80g_amdrx470-4", "2", "8 GiB", "80 GiB", "4"),
        ("GPU Instance 5x AMD RX470", "vcpu-2_memory-8g_disk-80g_amdrx470-5", "2", "8 GiB", "80 GiB", "5"),
        ("GPU Instance 6x AMD RX470", "vcpu-2_memory-8g_disk-80g_amdrx470-6", "2", "8 GiB", "80 GiB", "6"),
        ("GPU Instance 7x AMD RX470", "vcpu-2_memory-8g_disk-80g_amdrx470-7", "2", "8 GiB", "80 GiB", "7"),
        ("GPU Instance 8x AMD RX470", "vcpu-2_memory-8g_disk-80g_amdrx470-8", "2", "8 GiB", "80 GiB", "8"),
        ("CPU Instance 2x vCPU", "vcpu-2_memory-4g_disk-80g", "2", "4 GiB", "80 GiB", "0"),
        ("CPU Instance 4x vCPU", "vcpu-4_memory-8g_disk-80g", "4", "8 GiB", "80 GiB", "0"),
        ("CPU Instance 8x vCPU", "vcpu-8_memory-16g_disk-80g", "8", "16 GiB", "80 GiB", "0"),
        ("CPU Instance 12x vCPU", "vcpu-12_memory-24g_disk-80g", "12", "24 GiB", "80 GiB", "0"),
        ("CPU Instance 16x vCPU", "vcpu-16_memory-32g_disk-80g", "16", "32 GiB", "80 GiB", "0"),
        ("CPU Instance 20x vCPU", "vcpu-20_memory-40g_disk-80g", "20", "40 GiB", "80 GiB", "0"),
        ("CPU Instance 24x vCPU", "vcpu-24_memory-48g_disk-80g", "24", "48 GiB", "80 GiB", "0")]
    # Kludge because I didn't want to type in all the prices, which will change,
    # but currently obey linear scaling laws.
    for idx in range(1, len(data)):
        data[idx] = list(data[idx])
        product_name = data[idx][0]
        if "RTX 3090" in product_name:
            price_per_gpu = 1.70
        if "RTX 3080" in product_name:
            price_per_gpu = 1.10
        if "NVIDIA 1080Ti" in product_name:
            price_per_gpu = 0.60
        if "AMD MI25" in product_name:
            price_per_gpu = 0.80
        if "AMD RX470" in product_name:
            price_per_gpu = 0.36
        data[idx] += [f"${int(data[idx][-1])*price_per_gpu}"]
    return data
