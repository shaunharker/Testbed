# genesiscloud/offerings.py
# Shaun Harker, 2021-07-07

license = """
MIT LICENSE

Copyright (c) 2021 Shaun Harker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def instance_types():
    offerings = [
        ("Type (Product Name)", "API Instance Type Identifier", "vCPUs", "Memory", "Disk", "GPUs"),
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
