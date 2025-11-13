#pragma once
#include "types.hpp"
#include "device_data.hpp"

// Copy host CSR data to device-resident CSR (head/list).
DeviceCSR copy_csr_to_device(const CSR& host);

// Build half neighbor list (i<j) entirely on GPU using brute-force O(N^2/2),
// excluding 1-2 bonded pairs from device CSR adjacency. Returns device CSR.
DeviceCSR build_half_neighbors_gpu(const DeviceAtoms& atoms,
                                   const Box& box,
                                   const DeviceCSR& bonds12_dev,
                                   real rc);
