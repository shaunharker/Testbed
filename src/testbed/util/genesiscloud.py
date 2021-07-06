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
        assert_that(body_parameters, is_a_create_instance_body)
        response = self.api(verb=POST,
                            expected_status_code=201,
                            endpoint="https://api.genesiscloud.com/compute/v1/instance",
                            body_parameters=body_parameters)
        assert_that(response, is_a_create_instance_response)
        return response

    def list_all_instances(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#list-all-instances
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        assert_that(query_parameters, is_a_list_all_instances_query)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint="https://api.genesiscloud.com/compute/v1/instances",
                            query_parameters=query_parameters)
        assert_that(response, is_a_list_all_instances_response)
        return response

    def get_instance(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#get-instance
        """
        assert_that(instance_id, is_a_string())
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}")
        assert_that(response, is_a_get_instance_response)
        return response

    def destroy_an_instance(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#destroy-an-instance
        """
        # TODO: Consider an asynchronous version.
        # Refuse to destroy while instance is in "copying" state.
        assert_that(instance_id, is_an_instance_id())
        while self.get_instance(instance_id)["status"] == "copying":
            time.sleep(1.0)
        return self.api(verb=DELETE,
                        expected_status_code=204,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}")

    def snapshot_an_instance(self, instance_id, body_parameters, **kwargs): #snapshot_name):
        """
        https://developers.genesiscloud.com/instances#snapshot-an-instance
        """
        assert_that(instance_id, is_an_instance_id())
        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_a_snapshot_an_instance_body)
        response = self.api(verb=POST,
                            expected_status_code=201,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/snapshots",
                            body_parameters=body_parameters)
        assert_that(response, is_a_snapshot_an_instance_response)
        return response

    def list_snapshots_of_an_instance(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#list-snapshots-of-an-instance
        """
        assert_that(instance_id, is_an_instance_id()))
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/snapshots")
        assert_that(response, is_a_list_snapshots_of_an_instance_response)
        return response

    def attachdetach_security_groups_from_an_instance(self, instance_id, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#attachdetach-security-groups-from-an-instance
        """
        assert_that(instance_id, is_an_instance_id())
        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_an_attachdetach_security_groups_from_an_instance_body)
        response = self.api(verb=PATCH,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}",
                            body_parameters=body_parameters)
        assert_that(response, is_an_attachdetach_security_groups_from_an_instance_response)
        return response

    def attachdetach_volumes_from_an_instance(self, instance_id, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#attachdetach-volumes-from-an-instance
        """
        assert_that(instance_id, is_an_instance_id())
        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_an_attachdetach_volumes_from_an_instance_body)
        response = self.api(verb=PATCH,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}",
                            body_parameters=body_parameters)
        assert_that(response, is_an_attachdetach_volumes_from_an_instance_response)
        return response

    def update_an_instance(self, instance_id, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/instances#update-an-instance
        """
        assert_that(instance_id, is_an_instance_id())
        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_a_update_an_instance_body)
        response = self.api(verb=PATCH,
                        expected_status_code=200,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}",
                        body_parameters=body_parameters)
        assert_that(response, is_an_update_an_instance_response)
        return response

    def get_instance_actions(self, instance_id):
        """
        https://developers.genesiscloud.com/instances#get-instance-actions
        """
        assert_that(instance_id, is_an_instance_id())
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/actions")
        assert_that(response, is_a_get_instance_actions_response)
    def perform_action(self, instance_id, body_parameters=None, **kwarg):
        """
        https://developers.genesiscloud.com/instances#perform-action
        """
        assert_that(instance_id, is_an_instance_id())

        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_a_perform_action_body)
        response = self.api(verb=POST,
                            expected_status_code=204,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/instances/{instance_id}/actions",
                            body_parameters=body_parameters)
        assert_that(response, is_a_perform_action_response)
        return response

    # Images. See https://developers.genesiscloud.com/images
    def list_images(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/images#list-images
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        assert_that(query_parameters, is_a_list_images_query)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/images",
                            query_parameters=query_parameters)
        assert_that(response, is_a_list_images_response)
        return response

    # Snapshots. https://developers.genesiscloud.com/snapshots
    def list_snapshots(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/snapshots#list-snapshots
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        assert_that(query_parameters, is_a_list_snapshots_query)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/snapshots",
                            query_parameters=query_parameters)
        assert_that(response, is_a_list_snapshots_response)
        return response

    def get_snapshot(self, snapshot_id):
        """
        https://developers.genesiscloud.com/snapshots#get-snapshot
        """
        assert_that(instance_id, is_a_snapshot_id())
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/snapshots/{snapshot_id}")
        assert_that(response, is_a_get_snapshot_response)
        return response

    def delete_a_snapshot(self, snapshot_id):
        """
        https://developers.genesiscloud.com/snapshots#delete-a-snapshot
        """
        assert_that(instance_id, is_a_snapshot_id())
        return self.api(verb=DELETE,
                        expected_status_code=204,
                        endpoint=f"https://api.genesiscloud.com/compute/v1/snapshots/{snapshot_id}")

    # Volumes. See https://developers.genesiscloud.com/volumes
    def create_a_volume(self, body_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/volumes#create-a-volume
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_a_create_a_volume_body)
        response = self.api(verb=POST,
                            expected_status_code=201,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/volumes",
                            body_parameters=body_parameters)
        assert_that(response, is_a_create_a_volume_response)
        return response

    def list_volumes(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/volumes#list-volumes
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        assert_that(query_parameters, is_a_list_volumes_query)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/volumes",
                            query_parameters=query_parameters)
        assert_that(response, is_a_list_volumes_response)
        return response

    def get_volume(self, volume_id):
        """
        https://developers.genesiscloud.com/volumes#get-volume
        """
        assert_that(volume_id, is_a_volume_id)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/volumes/{volume_id}")
        assert_that(response, is_a_get_volume_response)
        return response

    def delete_a_volume(self, volume_id):
        """
        https://developers.genesiscloud.com/volumes#delete-a-volume
        """
        assert_that(volume_id, is_a_volume_id)
        response = self.api(verb=DELETE,
                            expected_status_code=204,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/volumes/{volume_id}")
        assert_that(response, is_a_delete_a_volume_response)
        return response

    # SSH Keys. See https://developers.genesiscloud.com/ssh-keys
    def list_ssh_keys(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/ssh-keys#list-ssh-keys
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        assert_that(query_parameters, is_a_list_ssh_keys_query)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/ssh-keys",
                            query_parameters=query_parameters)
        assert_that(response, is_a_list_ssh_keys_response)
        return response

    # Security Groups. See https://developers.genesiscloud.com/security-groups
    def create_security_groups(self, body_parameters, **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#create-security-groups
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_a_create_security_groups_body)
        response = self.api(verb=POST,
                            expected_status_code=201,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups",
                            body_parameters=body_parameters)
        assert_that(response, is_a_create_security_groups_response)
        return response

    def update_security_groups(self, body_parameters, **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#update-security-groups
        """
        body_parameters = self.parse_args(body_parameters, **kwargs)
        assert_that(body_parameters, is_an_update_security_groups_body)
        response = self.api(verb=PUT,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups",
                            body_parameters=body_parameters)
        assert_that(response, is_an_update_security_groups)
        return response

    def list_security_groups(self, query_parameters=None, **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#list-security-groups
        """
        query_parameters = self.parse_args(query_parameters, **kwargs)
        assert_that(query_parameters, is_a_list_security_groups_query)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups",
                            query_parameters=query_parameters)
        assert_that(response, is_a_list_security_groups_response)
        return response

    def get_security_group(self, security_group_id):
        """
        https://developers.genesiscloud.com/security-groups#get-security-group
        """
        assert_that(security_group_id, is_a_security_group_id)
        response = self.api(verb=GET,
                            expected_status_code=200,
                            endpoint=f"https://api.genesiscloud.com/compute/v1/security-groups/{security_group_id}")
        assert_that(response, is_a_get_security_group_response)
        return response

    def delete_a_security_group(self, security_group_id):
        """
        https://developers.genesiscloud.com/security-groups#delete-a-security-group
        """
        assert_that(security_group_id, is_a_security_group_id)
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


# We build up a rudimentary type-checking system
# in order to prevent sending erroneous API requests
# and also to make the code more self-documenting.

def assert_that(x, verifier):
    assert verifier(x)
    return True

def has_type(type):
    def verifier(value):
        assert type(value) is type
        return True
    return verifier

def is_a_string():
    return has_type(str)

strings = is_a_string

def is_an_integer():
    return has_type(int)

integers = is_an_integer

def is_an_id():
    return is_a_string()

ids = is_an_id

def is_an_ISO8601_time():
    return is_a_string()

ISO8601_times = is_an_ISO8601_time

def is_an_ip_address():
    return is_a_string()

ip_addresses = is_an_ip_address

def is_a_choice_from(*choices):
    def verifier(value):
        assert value in choices
        return True
    return verifier

choices_from = is_a_choice_from

def is_boolean():
    return is_a_choice_from("true", "false")

booleans = is_boolean

def is_a_list_of(item_type):
    list_verifier = has_type(list)
    item_verifier = has_type(item_type)
    def verifier(value):
        assert list_verifier(value)
        for item in value:
            assert item_verifier(item)
        return True
    return verifier

lists_of = is_a_list_of

def is_an_object(required={}, optional={}, **kwargs):
    # kwargs can be used for convenience to patch required
    for (k, v) in kwargs.items():
        required[k] = v
    def verifier(value):
        for (k, v) in required.items():
            assert k in value
            assert v(k)
        for (k, v) in optional.items():
            if k in value:
                assert v(k)
        for k in value:
            assert k in required or k in optional
        return True
    return verifier

objects = is_an_object

def is_an_api_instance_type_identifier():
    return is_a_string()

api_instance_type_identifiers = is_an_api_instance_type_identifier

def is_an_instance_id():
    return is_an_id()

instance_ids = is_an_instance_id

def is_an_image_id():
    return is_an_id()

image_ids = is_an_image_id

def is_a_snapshot_id():
    return is_an_id()

snapshot_ids = is_a_snapshot_id

def is_a_security_group_id():
    return is_an_id()

security_group_ids = is_a_security_group_id

def is_a_volume_id():
    return is_an_id()

volume_ids = is_a_volume_id

def is_an_instance():
    return is_an_object(
        required={
            "id": is_an_instance_id(),
            "name": is_a_string(),
            "hostname": is_a_string(),
            "type": is_an_api_instance_type_identifier(),
            "image": is_an_object({
                "id": is_an_image_id(),
                "name": is_a_string()}),
            "ssh_keys": is_a_list_of(objects({
                "id": is_an_ssh_key_id(),
                "name": is_a_string()})),
            "security_groups": is_a_list_of(objects({
                "id": is_a_security_group_id(),
                "name": is_a_string()})),
            "volumes": is_a_list_of(objects({
                "id": is_a_volume_id(),
                "name": is_a_string()})),
            "is_protected": is_a_boolean(),
            "status": is_a_choice_from("enqueued", "creating", "active",
                                       "shutdown", "copying", "restarting",
                                       "starting", "stopping", "deleting",
                                       "error", "unknown"),
            "created_at": is_an_ISO8601_time(),
            "updated_at": is_an_ISO8601_time()},
        optional={
            "private_ip": is_an_ip_address(),
            "public_ip": is_an_ip_address()})

instances = is_an_instance

def is_an_image():
    return is_an_object({
        "id": is_a_string(),
        "name": is_a_string(),
        "type": is_a_choice_from("base-os", "preconfigured", "snapshot"),
        "created_at": is_an_ISO8601_time()})

images = is_an_image

def is_a_snapshot():
    return is_an_object({
        "id": is_a_snapshot_id(),
        "name": is_a_string(),
        "status": is_a_choice_from("creating", "active", "shutdown",
                                   "copying", "restarting", "starting",
                                   "stopping", "deleting", "error",
                                   "unknown"),
        "resource_id": is_an_instance_id(),
        "created_at": is_an_ISO8601_time()})

snapshots = is_a_snapshot

def is_a_create_instance_body():
    return is_an_object(
        required={
            "name": is_a_string(),
            "hostname": is_a_string(),
            "type": is_an_api_instance_type_identifier(),
            "image": is_an_image_id(),
            "ssh_keys": is_a_list_of(ssh_key_ids())}},
        optional={
            "password": is_a_string(),
            "security_groups": is_a_list_of(security_group_ids()),
            "is_protected": is_a_boolean(),
            "metadata": is_an_object(
                optional={
                    "startup_script": is_a_bash_script()})})

def is_a_create_instance_response():
    return is_an_instance()

def is_a_list_all_instances_response():
    return is_an_object({
        "total_count": is_an_integer(),
        "page": is_an_integer(),
        "per_page": is_an_integer(),
        "instances": is_a_list_of(instances())})

def is_a_get_instance_response():
    return is_a_list_all_instances_response()

def is_a_snapshot_an_instance_response():
    return is_a_snapshot()

def is_a_list_snapshots_of_an_instance_response():
    return is_an_object({
        "total_count": is_an_integer(),
        "page": is_an_integer(),
        "per_page": is_an_integer(),
        "snapshots": is_a_list_of(snapshots())})

def is_an_attachdetach_security_groups_from_an_instance_body():
    return is_an_object({
        "security_groups": is_a_list_of(security_group_ids())})

def is_an_attachdetach_security_groups_from_an_instance_response():
    return is_an_instance()

def is_an_attachdetach_volumes_from_an_instance_body():
    return is_an_object({
        "volumes": is_a_list_of(volume_ids())})

def is_an_attachdetach_volumes_from_an_instance_response():
    return is_an_instance()

def is_an_update_an_instance_body():
    return is_an_object(
        optional={
            "name": is_a_string(),
            "is_protected": is_a_boolean(),})

def is_an_update_an_instance_response():
    return is_an_instance()

def is_a_get_instance_actions_response():
    return is_an_object({
        "actions": is_a_list_of(choices_from("start", "stop", "reset"))})

def is_a_perform_action_body():
    return is_an_object({
        "action": is_a_choice_from("start", "stop", "reset")})

def is_a_list_images_query():
    return is_an_object(optional={
        "type": is_a_choice_from("base-os", "preconfigured", "snapshot"),
        "page": is_an_integer(),
        "per_page": is_an_integer()})

def is_a_list_images_response():
    return is_an_object({
        "type": is_a_choice_from("base-os", "preconfigured", "snapshot"),
        "page": is_an_integer(),
        "per_page": is_an_integer(),
        "images": is_a_list_of(images())})

def is_a_list_snapshots_query():
    return is_an_object(
        optional={
            "per_page": is_an_integer(),
            "page": is_an_integer()})

def is_a_list_snapshots_response():
    return is_an_object({
        "snapshots": is_a_list_of(snapshots()),
        "total_count": is_an_integer(),
        "page": is_an_integer(),
        "per_page": is_an_integer()})

def is_a_get_snapshot_response():
    return is_a_snapshot()

def is_a_create_a_volume_body():
    return is_an_object({
        "name": is_a_string(),
        "description": is_a_string(),
        "size": is_an_integer()})

def is_a_volume():
    return is_an_object({
        "id": is_a_volume_id(),
        "name": is_a_string(),
        "description": is_a_string(),
        "size": is_an_integer(), # Note: genesiscloud makes an error in doc here?
        "instances": is_a_list_of(objects({
            "id": is_an_instance_id(),
            "name": is_a_string()})),
        "status": is_a_choice_from("available", "deleted", "error"),
        "created_at": is_an_ISO8601_time()})

volumes = is_a_volume

def is_a_create_a_volume_response():
    return is_a_volume()

def is_a_list_volumes_query():
    return is_an_object(
        optional={
            "per_page": is_an_integer(),
            "page": is_an_integer()})

def is_a_list_volumes_response():
    return is_an_object({
        "volumes": is_a_list_of(volumes()),
        "total_count": is_an_integer(),
        "page": is_an_integer(),
        "per_page": is_an_integer()})

def is_a_get_volume_response():
    return is_a_volume()

def is_a_list_ssh_keys_response():
    return is_an_object({
        "ssh_keys":
            is_a_list_of(objects({
                "id": is_a_string(),
                "name": is_a_string(),
                "public_key": is_a_string(),
                "created_at": is_an_ISO8601_time()})))),
        "total_count": is_an_integer(),
        "page": is_an_integer(),
        "per_page": is_an_integer()})

def is_a_rule():
    object_verifier = is_an_object(
        required={
            "protocol": choice_from("icmp", "tcp", "udp"),
            "direction": choice_from("ingress", "egress")}
        optional={
            "port_range_min": is_an_integer(),
            "port_range_max": is_an_integer()})
    extra_requirements = is_an_object(
        required={
            "port_range_min": is_an_integer(),
            "port_range_max": is_an_integer()})
    def verifier(value):
        assert object_verifier(value)
        if value["protocol"] in ["tcp", "udp"]:
            assert extra_requirements(value)
        return True
    return verifier

rules = is_a_rule

def is_a_security_group():
    return is_an_object({
        "id": is_a_string(),
        "name": is_a_string(),
        "description": is_a_string(),
        "status": is_a_choice_from("enqueued", "creating", "created",
                                   "updating", "deleting", "error"),
        "rules": is_a_list_of(rules()),
        "created_at": is_an_ISO8601_time()})

security_groups = is_a_security_group

def is_a_create_security_group_body():
    return is_an_object(
        required={
            "name": is_a_string(),
            "rules": is_a_list_of(rules())},
        optional={
            "description": is_a_string()})

def is_a_create_security_group_response():
    return is_a_security_group()

def is_an_update_security_groups_body():
    return is_an_object(
        required={
            "name": is_a_string(),
            "rules": is_a_list_of(rules())},
        optional={
            "description": is_a_string()})

def is_an_update_security_groups_response():
    return is_a_security_group()

def is_a_list_security_groups_query():
    return is_an_object(
        optional={
            "per_page": is_an_integer(),
            "page": is_an_integer()})

def is_a_list_security_groups_response():
    return is_an_object({
        "security_groups": is_a_list_of(objects({
            "id": is_a_string(),
            "name": is_a_string(),
            "description": is_a_string(),
            "rules": is_a_list_of(rules())})),
        "total_count": is_an_integer(),
        "page": is_an_integer(),
        "per_page": is_an_integer()})

def is_a_get_security_group_response():
    return is_an_object({
        "security_group": is_an_object({
            "id": is_a_string(),
            "name": is_a_string(),
            "description": is_a_string(),
            "rules": is_a_list_of(rules()),
            "created_at": is_an_ISO8601_time()})})
