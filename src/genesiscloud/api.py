# genesiscloud.py -- a schema checking python wrapper for api.genesiscloud.com
# Shaun Harker, 2021-07-05
#
# MIT LICENSE
#
# Copyright (c) 2021 Shaun Harker
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .validate import validate_input, validate_output

import json
import time
import os
from requests import get as GET
from requests import post as POST
from requests import put as PUT
from requests import patch as PATCH
from requests import delete as DELETE


class Client:
    """
    Client class for api.genesiscloud.com.
    To use this class one must first obtain API_TOKEN from
    https://account.genesiscloud.com/dashboard/security
    This API_TOKEN should be stored in an environment variable
    named GENESISCLOUD_API_TOKEN. This can usually be achieved
    by adding the line

        export GENESISCLOUD_API_TOKEN=<YOUR API_TOKEN HERE>

    to your `~/.bashrc` file.
    """
    def __init__(self):
        self.time_of_last_api_call = time.time()

    # Instances. See https://developers.genesiscloud.com/instances
    class CreateInstanceRequest(TypedDict):
        name: str
        hostname: str
        type: str
        image: str
        ssh_keys: List[str]


    class CreateInstanceRequestWithOptional(CreateInstanceRequest)
    def create_instance(self,
                        body_parameters: TypedDict={},
                        **kwargs):
        """
        https://developers.genesiscloud.com/instances#create-an-instance

        is_valid_input_to_create_instance = (
            is_an_object(
                required={
                    "name": is_a_string(),
                    "hostname": is_a_string(),
                    "type": is_an_api_instance_type_identifier(),
                    "image": is_an_image_id(),
                    "ssh_keys": is_a_list_of(ssh_key_ids())},
                optional={
                    "password": is_a_string(),
                    "security_groups": is_a_list_of(security_group_ids()),
                    "is_protected": is_a_boolean(),
                    "metadata": is_an_object(
                        optional={
                            "startup_script": is_a_bash_script()})}))

        is_valid_output_from_create_instance = (
                is_an_object(
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
                        "public_ip": is_an_ip_address()}))

        """
        response = self.api(
            command="create_instance",
            verb=POST,
            expected_status_code=201,
            endpoint="https://api.genesiscloud.com"
                     "/compute/v1/instance",
            body_parameters=body_parameters.update(kwargs))
        return response

    def list_all_instances(self,
                           query_parameters={},
                           **kwargs):
        """
        https://developers.genesiscloud.com/instances#list-all-instances
        """
        response = self.api(
            command="list_all_instances",
            verb=GET,
            expected_status_code=200,
            endpoint="https://api.genesiscloud.com"
                     "/compute/v1/instances",
            query_parameters=query_parameters.update(kwargs))
        return response

    def get_instance(self,
                     instance_id):
        """
        https://developers.genesiscloud.com/instances#get-instance
        """
        response = self.api(
            command="get_instance",
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}",
            instance_id=instance_id)
        return response

    def destroy_an_instance(self,
                            instance_id):
        """
        https://developers.genesiscloud.com/instances#destroy-an-instance
        """
        # TODO: Consider an asynchronous version.
        # Refuse to destroy while instance is in "copying" state.
        while self.get_instance(instance_id)["status"] == "copying":
            time.sleep(1.0)
        response = self.api(
            verb=DELETE,
            expected_status_code=204,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}",
            instance_id=instance_id)
        return response

    def snapshot_an_instance(self,
                             instance_id,
                             body_parameters={},
                             **kwargs):
        """
        https://developers.genesiscloud.com/instances#snapshot-an-instance
        """
        response = self.api(
            verb=POST,
            expected_status_code=201,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}/snapshots",
            instance_id=instance_id,
            body_parameters=body_parameters.update(kwargs))
        return response

    def list_snapshots_of_an_instance(self,
                                      instance_id):
        """
        https://developers.genesiscloud.com/instances#list-snapshots-of-an-instance
        """
        response = self.api(
            command="list_snapshots_of_an_instance",
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}/snapshots",
            instance_id=instance_id)
        return response

    def attachdetach_security_groups_from_an_instance(self,
                                                      instance_id,
                                                      body_parameters={},
                                                      **kwargs):
        """
        https://developers.genesiscloud.com/instances#attachdetach-security-groups-from-an-instance
        """
        response = self.api(
            command="attachdetach_security_groups_from_an_instance",
            verb=PATCH,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}",
            instance_id=instance_id,
            body_parameters=body_parameters.update(kwargs))
        return response

    def attachdetach_volumes_from_an_instance(self,
                                              instance_id,
                                              body_parameters={},
                                              **kwargs):
        """
        https://developers.genesiscloud.com/instances#attachdetach-volumes-from-an-instance
        """
        response = self.api(
            command="attachdetach_volumes_from_an_instance",
            verb=PATCH,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}",
            instance_id=instance_id,
            body_parameters=body_parameters.update(kwargs))
        return response

    def update_an_instance(self,
                           instance_id,
                           body_parameters={},
                           **kwargs):
        """
        https://developers.genesiscloud.com/instances#update-an-instance
        """
        response = self.api(
            verb=PATCH,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}",
            instance_id=instance_id,
            body_parameters=body_parameters.update(kwargs))
        return response

    def get_instance_actions(self,
                             instance_id):
        """
        https://developers.genesiscloud.com/instances#get-instance-actions
        """
        response = self.api(
            command="get_instance_actions",
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}/actions",
            instance_id=instance_id)
        return response

    def perform_action(self,
                       instance_id,
                       body_parameters={},
                       **kwargs):
        """
        https://developers.genesiscloud.com/instances#perform-action
        """
        response = self.api(
            verb=POST,
            expected_status_code=204,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/instances/{instance_id}/actions",
            instance_id=instance_id,
            body_parameters=body_parameters.update(kwargs))
        return response

    # Images. See https://developers.genesiscloud.com/images

    def list_images(self,
                    query_parameters={},
                    **kwargs):
        """
        https://developers.genesiscloud.com/images#list-images
        """
        response = self.api(
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/images",
            query_parameters=query_parameters.update(kwargs))
        return response

    # Snapshots. https://developers.genesiscloud.com/snapshots

    def list_snapshots(self,
                       query_parameters={},
                       **kwargs):
        """
        https://developers.genesiscloud.com/snapshots#list-snapshots
        """
        response = self.api(
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/snapshots",
            query_parameters=query_parameters.update(kwargs))
        return response

    def get_snapshot(self,
                     snapshot_id):
        """
        https://developers.genesiscloud.com/snapshots#get-snapshot
        """
        response = self.api(
            command="get_snapshot",
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/snapshots/{snapshot_id}",
            snapshot_id=snapshot_id)
        return response

    def delete_a_snapshot(self,
                          snapshot_id):
        """
        https://developers.genesiscloud.com/snapshots#delete-a-snapshot
        """
        response = self.api(
            command="delete_a_snapshot",
            verb=DELETE,
            expected_status_code=204,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/snapshots/{snapshot_id}",
            snapshot_id=snapshot_id)
        return response

    # Volumes. See https://developers.genesiscloud.com/volumes

    def create_a_volume(self,
                        body_parameters={},
                        **kwargs):
        """
        https://developers.genesiscloud.com/volumes#create-a-volume
        """
        response = self.api(
            verb=POST,
            expected_status_code=201,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/volumes",
            body_parameters=body_parameters.update(kwargs))
        return response

    def list_volumes(self,
                     query_parameters={},
                     **kwargs):
        """
        https://developers.genesiscloud.com/volumes#list-volumes
        """
        response = self.api(
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/volumes",
            query_parameters=query_parameters.update(kwargs))
        return response

    def get_volume(self,
                   volume_id):
        """
        https://developers.genesiscloud.com/volumes#get-volume
        """
        response = self.api(
            command="get_volume",
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/volumes/{volume_id}",
            volume_id=volume_id)
        return response

    def delete_a_volume(self,
                        volume_id):
        """
        https://developers.genesiscloud.com/volumes#delete-a-volume
        """
        response = self.api(
            command="delete_a_volume",
            verb=DELETE,
            expected_status_code=204,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/volumes/{volume_id}",
            volume_id=volume_id)
        return response

    # SSH Keys. See https://developers.genesiscloud.com/ssh-keys

    def list_ssh_keys(self,
                      query_parameters={},
                      **kwargs):
        """
        https://developers.genesiscloud.com/ssh-keys#list-ssh-keys
        """
        response = self.api(
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/ssh-keys",
            query_parameters=query_parameters.update(kwargs))
        return response

    # Security Groups. See https://developers.genesiscloud.com/security-groups

    def create_security_groups(self,
                               body_parameters={},
                               **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#create-security-groups
        """
        response = self.api(
            verb=POST,
            expected_status_code=201,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/security-groups",
            body_parameters=body_parameters.update(kwargs))
        return response

    def update_security_groups(self,
                               body_parameters={},
                               **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#update-security-groups
        """
        response = self.api(
            verb=PUT,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/security-groups",
            body_parameters=body_parameters.update(kwargs))
        return response

    def list_security_groups(self,
                             query_parameters={},
                             **kwargs):
        """
        https://developers.genesiscloud.com/security-groups#list-security-groups
        """
        response = self.api(
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/security-groups",
            query_parameters=query_parameters.update(kwargs))
        return response

    def get_security_group(self,
                           security_group_id):
        """
        https://developers.genesiscloud.com/security-groups#get-security-group
        """
        response = self.api(
            command="get_security_group",
            verb=GET,
            expected_status_code=200,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/security-groups/{security_group_id}",
            security_group_id=security_group_id)
        return response

    def delete_a_security_group(self,
                                security_group_id):
        """
        https://developers.genesiscloud.com/security-groups#delete-a-security-group
        """
        response = self.api(
            command="delete_a_security_group",
            verb=DELETE,
            expected_status_code=204,
            endpoint=f"https://api.genesiscloud.com"
                     f"/compute/v1/security-groups/{security_group_id}",
            security_group_id=security_group_id)
        return response

    def api(self,
            command,
            verb,
            expected_status_code,
            endpoint,
            query_parameters=None,
            body_parameters=None,
            **kwargs):
        """
        Call used by the other methods to interact with api.genesiscloud.com
        """
        # Validate the input
        validate_input(command,
                       query_parameters=query_parameters,
                       body_parameters=body_parameters,
                       **kwargs)

        # This ensures we won't hit the self.api rate limit of 10Hz,
        # unless class users break the singleton pattern:
        #  (see https://developers.genesiscloud.com/#rate-limiting)
        before = self.time_of_last_api_call
        now = time.time()
        if now - before < 0:
            before = now
        if now < before + 0.1:  # rate limit
            time.sleep(0.1 - (now - before))
        self.time_of_last_api_call = now

        # Acquire API_TOKEN
        API_TOKEN = os.getenv("GENESISCLOUD_API_TOKEN")
        if API_TOKEN is None:
            raise RuntimeError(
                "Expected an environment variable GENESISCLOUD_API_TOKEN.  \n"
                "To use this class one must first obtain an API_TOKEN from \n"
                "https://account.genesiscloud.com/dashboard/security       \n"
                "This API_TOKEN should be stored in an environment variable\n"
                "named GENESISCLOUD_API_TOKEN. This can usually be achieved\n"
                "by adding the line                                        \n"
                "                                                          \n"
                "    export GENESISCLOUD_API_TOKEN=<YOUR API_TOKEN HERE>   \n"
                "                                                          \n"
                "to your `~/.bashrc` file.")
        headers = {"Content-Type": "application/json",
                   "X-Auth-Token": API_TOKEN}

        # Make the API request
        response = verb(
            f"https://api.genesiscloud.com/compute/v1/{endpoint}",
            headers=headers,
            params=query_parameters,
            json=body_parameters)

        # Check for an error response
        if response.status_code != expected_status_code:
            message = json.dumps(response.json(), indent=4, sort_keys=True)
            raise RuntimeError(f"Status code {response.status_code} from "
                               f"api.genesiscloud.com with response of \n"
                               f"{message}")

        # Validate the output
        try:
            validate_output(command, response)
        except AssertionError:
            print("Warning: could not validate response "
                  "from api.genesiscloud.com")
        # Decode as JSON and return
        return response.json()
