import base64
import os

import pytest
import responses

from .conftest import make_client, uses_predictor


# @uses_predictor("input_none")
# def test_no_input(client, match):
#     resp = client.post("/predictions")
#     assert resp.status_code == 200
#     assert resp.json() == match({"status": "succeeded", "output": "foobar"})


# @uses_predictor("input_none")
# def test_missing_input(client, match):
#     """Check we support missing input fields for backwards compatibility"""
#     resp = client.post("/predictions", json={})
#     assert resp.status_code == 200
#     assert resp.json() == match({"status": "succeeded", "output": "foobar"})


# @uses_predictor("input_none")
# def test_empty_input(client, match):
#     """Check we support empty input fields for backwards compatibility"""
#     resp = client.post("/predictions", json={"input": {}})
#     assert resp.status_code == 200
#     assert resp.json() == match({"status": "succeeded", "output": "foobar"})


@uses_predictor("input_string")
def test_good_str_input(client, match):
    resp = client.post("/predictions", json={"instances": [{"text": "baz"}]})
    assert resp.status_code == 200
    assert resp.json() == match({"predictions": ["baz"]})


@uses_predictor("input_integer")
def test_good_int_input(client, match):
    resp = client.post("/predictions", json={"instances": [{"num": 3}]})
    assert resp.status_code == 200
    assert resp.json() == match({"predictions": [27]})
    resp = client.post("/predictions", json={"instances": [{"num": -3}]})
    assert resp.status_code == 200
    assert resp.json() == match({"predictions": [-27]})


@uses_predictor("input_integer")
def test_bad_int_input(client):
    resp = client.post("/predictions", json={"instances": [{"num": "foo"}]})
    assert resp.json() == {
        "detail": [
            {
                "loc": ["body", "instances", 0, "num"],
                "msg": "value is not a valid integer",
                "type": "type_error.integer",
            }
        ]
    }
    assert resp.status_code == 422


@uses_predictor("input_integer_default")
def test_default_int_input(client, match):
    resp = client.post("/predictions", json={"instances": [{}]})
    assert resp.status_code == 200
    assert resp.json() == match({"predictions": [25]})

    resp = client.post("/predictions", json={"instances": [{"num": 3}]})
    assert resp.status_code == 200
    assert resp.json() == match({"predictions": [9]})

# Not supported yet
# @uses_predictor("input_file")
# def test_file_input_data_url(client, match):
#     resp = client.post(
#         "/predictions",
#         json={
#             "input": {
#                 "file": "data:text/plain;base64,"
#                 + base64.b64encode(b"bar").decode("utf-8")
#             }
#         },
#     )
#     assert resp.json() == match({"output": "bar", "status": "succeeded"})
#     assert resp.status_code == 200


# Not supported yet
# @uses_predictor("input_file")
# def test_file_input_with_http_url(client, httpserver, match):
#     # Use a real HTTP server rather than responses as file fetching occurs on
#     # the other side of the Worker process boundary.
#     httpserver.expect_request("/foo.txt").respond_with_data("hello")
#     resp = client.post(
#         "/predictions",
#         json={"input": {"file": httpserver.url_for("/foo.txt")}},
#     )
#     assert resp.json() == match({"output": "hello", "status": "succeeded"})


# Not supported yet
# @uses_predictor("input_path_2")
# def test_file_input_with_http_url_error(client, httpserver, match):
#     httpserver.expect_request("/foo.txt").respond_with_data("haha", status=404)
#     resp = client.post(
#         "/predictions",
#         json={"input": {"path": httpserver.url_for("/foo.txt")}},
#     )
#     assert resp.json() == match({"status": "failed"})


# Not supported yet
# @uses_predictor("input_path")
# def test_path_input_data_url(client, match):
#     resp = client.post(
#         "/predictions",
#         json={
#             "input": {
#                 "path": "data:text/plain;base64,"
#                 + base64.b64encode(b"bar").decode("utf-8")
#             }
#         },
#     )
#     assert resp.json() == match({"output": "txt bar", "status": "succeeded"})
#     assert resp.status_code == 200


# Not supported yet
# @uses_predictor("input_path_2")
# def test_path_temporary_files_are_removed(client, match):
#     resp = client.post(
#         "/predictions",
#         json={
#             "input": {
#                 "path": "data:text/plain;base64,"
#                 + base64.b64encode(b"bar").decode("utf-8")
#             }
#         },
#     )
#     temporary_path = resp.json()["output"]
#     assert not os.path.exists(temporary_path)


# Not supported yet
# @responses.activate
# @uses_predictor("input_path")
# def test_path_input_with_http_url(client, match):
#     responses.add(responses.GET, "http://example.com/foo.txt", body="hello")
#     resp = client.post(
#         "/predictions",
#         json={"input": {"path": "http://example.com/foo.txt"}},
#     )
#     assert resp.json() == match({"output": "txt hello", "status": "succeeded"})


# Not supported yet
# @uses_predictor("input_file")
# def test_file_bad_input(client):
#     resp = client.post(
#         "/predictions",
#         json={"input": {"file": "foo"}},
#     )
#     assert resp.status_code == 422


# Not supported yet
# @uses_predictor("input_multiple")
# def test_multiple_arguments(client, match):
#     resp = client.post(
#         "/predictions",
#         json={
#             "input": {
#                 "text": "baz",
#                 "num1": 5,
#                 "path": "data:text/plain;base64,"
#                 + base64.b64encode(b"wibble").decode("utf-8"),
#             }
#         },
#     )
#     assert resp.status_code == 200
#     assert resp.json() == match({"output": "baz 50 wibble", "status": "succeeded"})


@uses_predictor("input_ge_le")
def test_gt_lt(client):
    resp = client.post("/predictions", json={"instances": [{"num": 2}]})
    assert resp.json() == {
        "detail": [
            {
                "ctx": {"limit_value": 3.01},
                "loc": ["body", "instances", 0, "num"],
                "msg": "ensure this value is greater than or equal to 3.01",
                "type": "value_error.number.not_ge",
            }
        ]
    }
    assert resp.status_code == 422

    resp = client.post("/predictions", json={"instances": [{"num": 5}]})
    assert resp.status_code == 200


@uses_predictor("input_choices")
def test_choices_str(client):
    resp = client.post("/predictions", json={"instances": [{"text": "foo"}]})
    assert resp.status_code == 200
    resp = client.post("/predictions", json={"instances": [{"text": "baz"}]})
    assert resp.status_code == 422


@uses_predictor("input_choices_integer")
def test_choices_int(client):
    resp = client.post("/predictions", json={"instances": [{"x": 1}]})
    assert resp.status_code == 200
    resp = client.post("/predictions", json={"instances": [{"x": 3}]})
    assert resp.status_code == 422


def test_untyped_inputs():
    with pytest.raises(TypeError):
        make_client("input_untyped")


# def test_input_with_unsupported_type():
#     with pytest.raises(TypeError):
#         make_client("input_unsupported_type")
