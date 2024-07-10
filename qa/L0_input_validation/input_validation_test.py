#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

sys.path.append("../common")

import unittest

import infer_util as iu
import numpy as np
import tritonclient.grpc as tritongrpcclient
import tritonclient.http as tritonhttpclient
import tritonclient.utils as utils
from tritonclient.utils import (
    InferenceServerException,
    np_to_triton_dtype,
    shared_memory,
)


class InputValTest(unittest.TestCase):
    def test_input_validation_required_empty(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_all_required",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected 3 inputs but got 0 inputs for model 'input_all_required'. Got input(s) [], but missing required input(s) ['INPUT0','INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_optional_empty(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_optional",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected number of inputs between 3 and 4 but got 0 inputs for model 'input_optional'. Got input(s) [], but missing required input(s) ['INPUT0','INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_required_missing(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT0", [1], "FP32"))

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_all_required",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected 3 inputs but got 1 inputs for model 'input_all_required'. Got input(s) ['INPUT0'], but missing required input(s) ['INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_optional(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT0", [1], "FP32"))
        # Option Input is added, 2 required are missing

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_optional",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected number of inputs between 3 and 4 but got 1 inputs for model 'input_optional'. Got input(s) ['INPUT0'], but missing required input(s) ['INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_all_optional(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        result = triton_client.infer(
            model_name="input_all_optional",
            inputs=inputs,
        )
        response = result.get_response()
        self.assertIn(str(response.outputs[0].name), "OUTPUT0")


class InputShapeTest(unittest.TestCase):
    def test_client_input_shape_validation(self):
        model_name = "simple"

        for client_type in ["http", "grpc"]:
            if client_type == "http":
                triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
            else:
                triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

            # Infer
            inputs = []
            if client_type == "http":
                inputs.append(tritonhttpclient.InferInput("INPUT0", [1, 16], "INT32"))
                inputs.append(tritonhttpclient.InferInput("INPUT1", [1, 16], "INT32"))
            else:
                inputs.append(tritongrpcclient.InferInput("INPUT0", [1, 16], "INT32"))
                inputs.append(tritongrpcclient.InferInput("INPUT1", [1, 16], "INT32"))

            # Create the data for the two input tensors. Initialize the first
            # to unique integers and the second to all ones.
            input0_data = np.arange(start=0, stop=16, dtype=np.int32)
            input0_data = np.expand_dims(input0_data, axis=0)
            input1_data = np.ones(shape=(1, 16), dtype=np.int32)

            # Initialize the data
            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            # Compromised input shapes
            inputs[0].set_shape([2, 8])
            inputs[1].set_shape([2, 8])

            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(model_name=model_name, inputs=inputs)
            err_str = str(e.exception)
            self.assertIn(
                f"unexpected shape for input 'INPUT1' for model 'simple'. Expected [-1,16], got [2,8]",
                err_str,
            )

            # Compromised input shapes
            inputs[0].set_shape([1, 8])
            inputs[1].set_shape([1, 8])

            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(model_name=model_name, inputs=inputs)
            err_str = str(e.exception)
            self.assertIn(
                f"input 'INPUT0' got unexpected elements count 16, expected 8",
                err_str,
            )

    def test_client_input_string_shape_validation(self):
        for client_type in ["http", "grpc"]:

            def identity_inference(triton_client, np_array, binary_data):
                model_name = "simple_identity"

                # Total elements no change
                inputs = []
                if client_type == "http":
                    inputs.append(
                        tritonhttpclient.InferInput("INPUT0", np_array.shape, "BYTES")
                    )
                    inputs[0].set_data_from_numpy(np_array, binary_data=binary_data)
                    inputs[0].set_shape([2, 8])
                else:
                    inputs.append(
                        tritongrpcclient.InferInput("INPUT0", np_array.shape, "BYTES")
                    )
                    inputs[0].set_data_from_numpy(np_array)
                    inputs[0].set_shape([2, 8])
                triton_client.infer(model_name=model_name, inputs=inputs)

                # Compromised input shape
                inputs[0].set_shape([1, 8])

                with self.assertRaises(InferenceServerException) as e:
                    triton_client.infer(model_name=model_name, inputs=inputs)
                err_str = str(e.exception)
                self.assertIn(
                    f"input 'INPUT0' got unexpected elements count 16, expected 8",
                    err_str,
                )

            if client_type == "http":
                triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
            else:
                triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

            # Example using BYTES input tensor with utf-8 encoded string that
            # has an embedded null character.
            null_chars_array = np.array(
                ["he\x00llo".encode("utf-8") for i in range(16)], dtype=np.object_
            )
            null_char_data = null_chars_array.reshape([1, 16])
            identity_inference(triton_client, null_char_data, True)  # Using binary data
            identity_inference(triton_client, null_char_data, False)  # Using JSON data

            # Example using BYTES input tensor with 16 elements, where each
            # element is a 4-byte binary blob with value 0x00010203. Can use
            # dtype=np.bytes_ in this case.
            bytes_data = [b"\x00\x01\x02\x03" for i in range(16)]
            np_bytes_data = np.array(bytes_data, dtype=np.bytes_)
            np_bytes_data = np_bytes_data.reshape([1, 16])
            identity_inference(triton_client, np_bytes_data, True)  # Using binary data
            identity_inference(triton_client, np_bytes_data, False)  # Using JSON data

    def test_client_input_shm_size_validation(self):
        # We use a simple model that takes 2 input tensors of 16 integers
        # each and returns 2 output tensors of 16 integers each. One
        # output tensor is the element-wise sum of the inputs and one
        # output is the element-wise difference.
        model_name = "simple"

        for client_type in ["http", "grpc"]:
            if client_type == "http":
                triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
            else:
                triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
            # To make sure no shared memory regions are registered with the
            # server.
            triton_client.unregister_system_shared_memory()
            triton_client.unregister_cuda_shared_memory()

            # Create the data for the two input tensors. Initialize the first
            # to unique integers and the second to all ones.
            input0_data = np.arange(start=0, stop=16, dtype=np.int32)
            input1_data = np.ones(shape=16, dtype=np.int32)

            input_byte_size = input0_data.size * input0_data.itemsize

            # Create shared memory region for input and store shared memory handle
            shm_ip_handle = shared_memory.create_shared_memory_region(
                "input_data", "/input_simple", input_byte_size * 2
            )

            # Put input data values into shared memory
            shared_memory.set_shared_memory_region(shm_ip_handle, [input0_data])
            shared_memory.set_shared_memory_region(
                shm_ip_handle, [input1_data], offset=input_byte_size
            )

            # Register shared memory region for inputs with Triton Server
            triton_client.register_system_shared_memory(
                "input_data", "/input_simple", input_byte_size * 2
            )

            # Set the parameters to use data from shared memory
            inputs = []
            if client_type == "http":
                inputs.append(tritonhttpclient.InferInput("INPUT0", [1, 16], "INT32"))
                inputs.append(tritonhttpclient.InferInput("INPUT1", [1, 16], "INT32"))
            else:
                inputs.append(tritongrpcclient.InferInput("INPUT0", [1, 16], "INT32"))
                inputs.append(tritongrpcclient.InferInput("INPUT1", [1, 16], "INT32"))
            inputs[-2].set_shared_memory("input_data", input_byte_size + 4)
            inputs[-1].set_shared_memory(
                "input_data", input_byte_size, offset=input_byte_size
            )

            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(model_name=model_name, inputs=inputs)
            err_str = str(e.exception)
            self.assertIn(
                f"input 'INPUT0' got unexpected byte size {input_byte_size+4}, expected {input_byte_size}",
                err_str,
            )

            # Set the parameters to use data from shared memory
            inputs[-2].set_shared_memory("input_data", input_byte_size)
            inputs[-1].set_shared_memory(
                "input_data", input_byte_size - 4, offset=input_byte_size
            )

            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(model_name=model_name, inputs=inputs)
            err_str = str(e.exception)
            self.assertIn(
                f"input 'INPUT1' got unexpected byte size {input_byte_size-4}, expected {input_byte_size}",
                err_str,
            )

            print(triton_client.get_system_shared_memory_status())
            triton_client.unregister_system_shared_memory()
            assert len(shared_memory.mapped_shared_memory_regions()) == 1
            shared_memory.destroy_shared_memory_region(shm_ip_handle)
            assert len(shared_memory.mapped_shared_memory_regions()) == 0

    def test_client_input_string_shm_size_validation(self):
        # We use a simple model that takes 2 input tensors of 16 strings
        # each and returns 2 output tensors of 16 strings each. The input
        # strings must represent integers. One output tensor is the
        # element-wise sum of the inputs and one output is the element-wise
        # difference.
        model_name = "simple_string"

        for client_type in ["http", "grpc"]:
            if client_type == "http":
                triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
            else:
                triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

            # To make sure no shared memory regions are registered with the
            # server.
            triton_client.unregister_system_shared_memory()
            triton_client.unregister_cuda_shared_memory()

            # Create the data for the two input tensors. Initialize the first
            # to unique integers and the second to all ones.
            in0 = np.arange(start=0, stop=16, dtype=np.int32)
            in0n = np.array(
                [str(x).encode("utf-8") for x in in0.flatten()], dtype=object
            )
            input0_data = in0n.reshape(in0.shape)
            in1 = np.ones(shape=16, dtype=np.int32)
            in1n = np.array(
                [str(x).encode("utf-8") for x in in1.flatten()], dtype=object
            )
            input1_data = in1n.reshape(in1.shape)

            input0_data_serialized = utils.serialize_byte_tensor(input0_data)
            input1_data_serialized = utils.serialize_byte_tensor(input1_data)
            input0_byte_size = utils.serialized_byte_size(input0_data_serialized)
            input1_byte_size = utils.serialized_byte_size(input1_data_serialized)

            # Create Input0 and Input1 in Shared Memory and store shared memory handles
            shm_ip0_handle = shared_memory.create_shared_memory_region(
                "input0_data", "/input0_simple", input0_byte_size
            )
            shm_ip1_handle = shared_memory.create_shared_memory_region(
                "input1_data", "/input1_simple", input1_byte_size
            )

            # Put input data values into shared memory
            shared_memory.set_shared_memory_region(
                shm_ip0_handle, [input0_data_serialized]
            )
            shared_memory.set_shared_memory_region(
                shm_ip1_handle, [input1_data_serialized]
            )

            # Register Input0 and Input1 shared memory with Triton Server
            triton_client.register_system_shared_memory(
                "input0_data", "/input0_simple", input0_byte_size
            )
            triton_client.register_system_shared_memory(
                "input1_data", "/input1_simple", input1_byte_size
            )

            # Set the parameters to use data from shared memory
            inputs = []
            if client_type == "http":
                inputs.append(tritonhttpclient.InferInput("INPUT0", [1, 16], "BYTES"))
                inputs.append(tritonhttpclient.InferInput("INPUT1", [1, 16], "BYTES"))
            else:
                inputs.append(tritongrpcclient.InferInput("INPUT0", [1, 16], "BYTES"))
                inputs.append(tritongrpcclient.InferInput("INPUT1", [1, 16], "BYTES"))
            inputs[-2].set_shared_memory("input0_data", input0_byte_size + 4)
            inputs[-1].set_shared_memory("input1_data", input1_byte_size)

            with self.assertRaises(InferenceServerException) as e:
                triton_client.infer(model_name=model_name, inputs=inputs)
            err_str = str(e.exception)

            # BYTES inputs in shared memory will skip the check at the client
            self.assertIn(
                f"Invalid offset + byte size for shared memory region: 'input0_data'",
                err_str,
            )

            print(triton_client.get_system_shared_memory_status())
            triton_client.unregister_system_shared_memory()
            assert len(shared_memory.mapped_shared_memory_regions()) == 2
            shared_memory.destroy_shared_memory_region(shm_ip0_handle)
            shared_memory.destroy_shared_memory_region(shm_ip1_handle)
            assert len(shared_memory.mapped_shared_memory_regions()) == 0


if __name__ == "__main__":
    unittest.main()
