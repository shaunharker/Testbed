# validate.py
# Shaun Harker, 2021-07-05

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

# In this file we build up a rudimentary type-checking system
# from scratch and apply it to the problem of validating interactions
# with the API.
#
# Disclaimer: I am not claiming this is the best way to go about writing this
#  particular piece of code. Probably using JSON-schema checking tools would
#  be appropriate.
#
# But I was too lazy to do it the smart way, and this seemed like more fun.


def validate_input(function_name, **kwargs):
    if function_name == 'create_instance':
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_create_instance)
    if function_name == 'delete_a_security_group':
        assert_that(kwargs["security_group_id"],
                    String)
    if function_name == 'list_all_instances':
        assert_that(kwargs["query_parameters"],
                    is_valid_input_to_list_all_instances)
    if function_name == 'get_instance':
        assert_that(kwargs["instance_id"],
                    String)
    if function_name == 'destroy_an_instance':
        assert_that(kwargs["instance_id"],
                    String)
    if function_name == 'snapshot_an_instance':
        assert_that(kwargs["instance_id"],
                    String)
    if function_name == 'list_snapshots_of_an_instance':
        assert_that(kwargs["instance_id"],
                    is_valid_input_to_list_snapshots_of_an_instance)
    if function_name == 'attachdetach_security_groups_from_an_instance':
        assert_that(kwargs["instance_id"], String)
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_attachdetach_security_groups_from_an_instance) # noqa
    if function_name == 'attachdetach_volumes_from_an_instance':
        assert_that(kwargs["instance_id"], String)
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_attachdetach_volumes_from_an_instance)
    if function_name == 'update_an_instance':
        assert_that(kwargs["instance_id"], String)
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_update_an_instance)
    if function_name == 'get_instance_actions':
        assert_that(kwargs["instance_id"], String)
    if function_name == 'perform_action':
        assert_that(kwargs["instance_id"], String)
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_perform_action)
    if function_name == 'list_images':
        assert_that(kwargs["query_parameters"],
                    is_valid_input_to_list_images)
    if function_name == 'list_snapshots':
        assert_that(kwargs["query_parameters"],
                    is_valid_input_to_list_snapshots)
    if function_name == 'get_snapshot':
        assert_that(kwargs["snapshot_id"], String)
        assert_that(kwargs["query_parameters"],
                    is_valid_input_to_get_snapshot)
    if function_name == 'delete_a_snapshot':
        assert_that(kwargs["snapshot_id"], String)
    if function_name == 'create_a_volume':
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_create_a_volume)
    if function_name == 'list_volumes':
        assert_that(kwargs["query_parameters"],
                    is_valid_input_to_list_volumes)
    if function_name == 'get_volume':
        assert_that(kwargs["volume_id"], String)
    if function_name == 'delete_a_volume':
        assert_that(kwargs["volume_id"], String)
    if function_name == 'list_ssh_keys':
        assert_that(kwargs["query_parameters"],
                    is_valid_input_to_list_ssh_keys)
    if function_name == 'create_security_groups':
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_create_security_groups)
    if function_name == 'update_security_groups':
        assert_that(kwargs["body_parameters"],
                    is_valid_input_to_update_security_groups)
    if function_name == 'list_security_groups':
        assert_that(kwargs["query_parameters"],
                    is_valid_input_to_list_security_groups)
    if function_name == 'get_security_group':
        assert_that(kwargs["security_group_id"],
                    String)
    if function_name == 'delete_a_security_group':
        assert_that(kwargs["security_group_id"],
                    String)
    return True


def validate_output(function_name, **kwargs):
    if function_name == 'create_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_create_instance)
    if function_name == 'delete_a_security_group':
        assert_that(kwargs["response"],
                    String)
    if function_name == 'list_all_instances':
        assert_that(kwargs["response"],
                    is_valid_output_from_list_all_instances)
    if function_name == 'get_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_get_instance)
    if function_name == 'destroy_an_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_destroy_an_instance)
    if function_name == 'snapshot_an_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_snapshot_an_instance)
    if function_name == 'list_snapshots_of_an_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_list_snapshots_of_an_instance)
    if function_name == 'attachdetach_security_groups_from_an_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_attachdetach_security_groups_from_an_instance) # noqa
    if function_name == 'attachdetach_volumes_from_an_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_attachdetach_volumes_from_an_instance)
    if function_name == 'update_an_instance':
        assert_that(kwargs["response"],
                    is_valid_output_from_update_an_instance)
    if function_name == 'get_instance_actions':
        assert_that(kwargs["response"],
                    is_valid_output_from_get_instance_actions)
    if function_name == 'perform_action':
        assert_that(kwargs["response"],
                    is_valid_output_from_perform_action)
    if function_name == 'list_images':
        assert_that(kwargs["response"],
                    is_valid_output_from_list_images)
    if function_name == 'list_snapshots':
        assert_that(kwargs["response"],
                    is_valid_output_from_list_snapshots)
    if function_name == 'get_snapshot':
        assert_that(kwargs["response"],
                    is_valid_output_from_get_snapshot)
    if function_name == 'delete_a_snapshot':
        assert_that(kwargs["response"],
                    is_valid_output_from_delete_a_snapshot)
    if function_name == 'create_a_volume':
        assert_that(kwargs["response"],
                    is_valid_output_from_create_a_volume)
    if function_name == 'list_volumes':
        assert_that(kwargs["response"],
                    is_valid_output_from_list_volumes)
    if function_name == 'get_volume':
        assert_that(kwargs["response"],
                    is_valid_output_from_get_volume)
    if function_name == 'delete_a_volume':
        assert_that(kwargs["response"],
                    is_valid_output_from_delete_a_volume)
    if function_name == 'delete_a_snapshot':
        assert_that(kwargs["response"],
                    is_valid_output_from_delete_a_snapshot)
    if function_name == 'list_ssh_keys':
        assert_that(kwargs["response"],
                    is_valid_output_from_list_ssh_keys)
    if function_name == 'create_security_groups':
        assert_that(kwargs["response"],
                    is_valid_output_from_create_security_groups)
    if function_name == 'update_security_groups':
        assert_that(kwargs["response"],
                    is_valid_output_from_update_security_groups)
    if function_name == 'list_security_groups':
        assert_that(kwargs["response"],
                    is_valid_output_from_list_security_groups)
    if function_name == 'get_security_group':
        assert_that(kwargs["response"],
                    is_valid_output_from_get_security_group)
    if function_name == 'delete_a_security_group':
        assert_that(kwargs["response"],
                    is_valid_output_from_delete_a_security_group)
    return True


class Type:
    def __init__(self, constructor, recognizer):
        self.constructor = constructor
        self.recognizer = recognizer
    def __call__(self, *args, **kwargs):
        constructed_item = self.constructor(*args, **kwargs)
        assert constructed_item in self
        return constructed_item
    def __contains__(self, x):
        return self.recognizer(x)

TypeWrapper = lambda T: Type(constructor = lambda *args, **kwargs: T(*args, **kwargs),
                             recognizer = type(x) is T)
String = TypeWrapper(str)
Integer = TypeWrapper(int)
List = TypeWrapper(list)
Dict = TypeWrapper(dict)
TypedList = (
    lambda T:
        Type(constructor=List.constructor,
             recognizer=lambda L: L in List and
                                  all(x in T for x in L)))
MinDict = (
    lambda minimum:
        Type(constructor=Dict.constructor,
             recognizer=lambda D: D in Dict and
                                  all(k in D and D[k] in v
                                      for (k,v) in minimum.items())))
MaxDict = (
    lambda minimum:
        Type(constructor=Dict.constructor,
             recognizer=lambda D: D in Dict and
                                  all(k not in D or D[k] in v
                                  for (k,v) in maximum.items())))
TypedDict = (
    lambda required, optional={}:
        Type(constructor=Dict.constructor,
             recognizer=lambda D: D in MinDict(required) and
                                  D in MaxDict({**required,**optional})))
StringLiteral = (
    lambda *choices:
        Type(constructor=String.constructor,
             recognizer=lambda s: s in String and s in choices))
Boolean = StringLiteral("true", "false")
ISO8601_Time = String
IP_Address = String
Instance = (
    TypedDict(
        required={
            "id": String,
            "name": String,
            "hostname": String,
            "type": String,
            "image": TypedDict({
                "id": String,
                "name": String}),
            "ssh_keys": TypedList(TypedDict({
                "id": String,
                "name": String})),
            "security_groups": TypedList(TypedDict({
                "id": String,
                "name": String})),
            "volumes": TypedList(TypedDict({
                "id": String,
                "name": String})),
            "is_protected": Boolean,
            "status": StringLiteral("enqueued", "creating", "active",
                                    "shutdown", "copying", "restarting",
                                    "starting", "stopping", "deleting",
                                    "error", "unknown"),
            "created_at": ISO8601_Time,
            "updated_at": ISO8601_Time},
        optional={
            "private_ip": IP_Address,
            "public_ip": IP_Address}))


Image = (
    TypedDict({
        "id": String,
        "name": String,
        "type": Literal("base-os", "preconfigured", "snapshot"),
        "created_at": ISO8601_Time}))


Snapshot = (
    TypedDict({
        "id": String,
        "name": String,
        "status": Literal("creating", "active", "shutdown",
                          "copying", "restarting", "starting",
                          "stopping", "deleting", "error",
                          "unknown"),
        "resource_id": String,
        "created_at": ISO8601_Time}))


Volume = (
    TypedDict({
        "id": String,
        "name": String,
        "description": String,
        "size": Integer,  # c.f. genesiscloud docs
        "instances": TypedList(TypedDict({
            "id": String,
            "name": String})),
        "status": Literal("available", "deleted", "error"),
        "created_at": ISO8601_Time}))


Rule = lambda x: (
    TypedDict(
        required={
            "protocol": Literal("icmp", "tcp", "udp"),
            "direction": Literal("ingress", "egress")
        optional={
            "port_range_min": Integer, # or null?
            "port_range_max": Integer})(x) # or null?
    and Truth(
        x["protocol"] not in ["tcp", "udp"]
        or
        TypedDict(required={
            "port_range_min": Integer, # or null?
            "port_range_max": Integer})(x))) # or null?


SecurityGroup = (
    TypedDict({
        "id": String,
        "name": String,
        "description": String,
        "status": Literal("enqueued", "creating", "created",
                          "updating", "deleting", "error"),
        "rule": TypedList(Rule),
        "created_at": ISO8601_Time}))


CreateInstanceInput = (
    TypedDict(
        required={
            "name": String,
            "hostname": String,
            "type": String,
            "image": String,
            "ssh_keys": String)},
        optional={
            "password": String,
            "security_groups": String),
            "is_protected": Boolean,
            "metadata": TypedDict(
                optional={
                    "startup_script": String})}))


PaginatedInput = (
    TypedDict({
        "page": Integer,
        "per_page": Integer}))


PaginatedOutput = lambda name, T: (
    TypedDict({
        "total_count": Integer,
        "page": Integer,
        "per_page": Integer,
        name: TypedList(T)}))

is_valid_output_from_create_instance = Instance

is_valid_output_from_list_all_instances = PaginatedOutput("instances", Instance)

is_valid_output_from_get_instance = Instance

is_valid_output_from_snapshot_an_instance = Snapshot

is_valid_output_from_list_snapshots_of_an_instance = PaginatedOutput("snapshots", Snapshot)

is_valid_input_to_attachdetach_security_groups_from_an_instance = (
    TypedDict({"security_groups": String)}))

is_valid_output_from_attachdetach_security_groups_from_an_instance = Instance

is_valid_input_to_attachdetach_volumes_from_an_instance = (
    TypedDict({"volumes": String}))

is_valid_output_from_attachdetach_volumes_from_an_instance = Instance

is_valid_input_to_update_an_instance = (
    TypedDict(
        optional={
            "name": String,
            "is_protected": Boolean}))

is_valid_output_from_update_an_instance = Instance

is_valid_output_from_get_instance_actions = (
    TypedDict({
        "actions": TypedList(Literal("start", "stop", "reset"))}))

is_valid_input_to_perform_action = (
    TypedDict({
        "action": Literal("start", "stop", "reset")}))

is_valid_input_to_list_images = (
    TypedDict(optional={
        "type": Literal("base-os", "preconfigured", "snapshot"),
        "page": Integer,
        "per_page": Integer}))

is_valid_output_from_list_images = PaginatedOutput("images", Image)

is_valid_input_to_list_snapshots = PaginatedInput

is_valid_output_from_list_snapshots = PaginatedOutput("snapshots", Snapshot)

is_valid_output_from_get_snapshot = Snapshot

is_valid_input_to_create_a_volume = (
    TypedDict({
        "name": String,
        "description": String,
        "size": Integer}))

is_valid_output_from_create_a_volume = Volume

is_valid_input_to_list_volumes = PaginatedInput

is_valid_output_from_list_volumes = PaginatedOutput("volumes", Volume)

is_valid_output_from_get_volume = Volume

is_valid_output_from_list_ssh_keys = (
    PaginatedOutput("ssh_keys",
        TypedDict({
            "id": String,
            "name": String,
            "public_key": String,
            "created_at": ISO8601_Time})))


is_valid_input_to_create_security_group = (
    TypedDict(
        required={
            "name": String,
            "rule": TypedList(Rule)},
        optional={
            "description": String}))


is_valid_output_from_create_security_group = SecurityGroup

#

is_valid_input_to_update_security_groups = (
    TypedDict(
        required={
            "name": String,
            "rule": TypedList(Rule)},
        optional={
            "description": String}))


is_valid_output_from_update_security_groups = SecurityGroup

#

is_valid_input_to_list_security_groups = PaginationInput

is_valid_output_from_list_security_groups = (
    PaginationOutput("security_groups",
        TypedDict({
            "id": String,
            "name": String,
            "description": String,
            "rule": TypedList(Rule)})))

#

is_valid_output_from_get_security_group = (
    TypedDict({
        "security_group": TypedDict({
            "id": String,
            "name": String,
            "description": String,
            "rule": TypedList(Rule),
            "created_at": ISO8601_Time})}))
