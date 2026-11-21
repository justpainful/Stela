// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2011-2014, Intel Corporation.
//! The host cost of guest-sized transfers through the owned driver and its
//! model controller: the commands they take and the time on the host's
//! clock. The model completes each command at its doorbell, so the time is
//! the driver's and the model's work alone, no device latency.
use super::*;

/// The guest's largest storage transfer (`stela_soc` ANS `MAX_TRANSFER`).
const GUEST_TRANSFER: usize = 1024 * 1024;
const DISK_BYTES: usize = 64 << 20;
const ROUNDS: usize = 4;

/// `cargo test -p stela-driver-storage --release --lib transfer_cost --
/// --ignored --nocapture`
#[test]
#[ignore = "manual: opt-in"]
fn transfer_cost_of_guest_sized_reads_and_writes() {
    let f = Fixture::new();
    {
        let mut m = f.model.lock().unwrap();
        m.disk = vec![0; DISK_BYTES];
        m.blocks = (DISK_BYTES / 512) as u64;
    }
    let mut attached = f.attach().unwrap();
    let disk = &mut attached.namespaces[0];
    let transfers = DISK_BYTES / GUEST_TRANSFER;
    let data: Vec<u8> = (0..GUEST_TRANSFER).map(|n| (n * 31) as u8).collect();
    let mut buffer = vec![0u8; GUEST_TRANSFER];
    for (name, write) in [("write", true), ("read", false)] {
        let mut best = u128::MAX;
        let mut commands = 0;
        for _ in 0..ROUNDS {
            let before = f.model.lock().unwrap().commands;
            let started = std::time::Instant::now();
            for index in 0..transfers {
                let lba = (index * GUEST_TRANSFER / 512) as u64;
                if write {
                    disk.write(lba, &data).unwrap();
                } else {
                    disk.read(lba, &mut buffer).unwrap();
                }
            }
            best = best.min(started.elapsed().as_nanos());
            commands = f.model.lock().unwrap().commands - before;
        }
        if !write {
            assert_eq!(buffer, data);
        }
        std::println!(
            "TRANSFER {name}: max_transfer={}KiB commands_per_MiB={} us_per_MiB={:.1}",
            disk.identity().features.max_transfer >> 10,
            commands as f64 / transfers as f64,
            best as f64 / 1000.0 / transfers as f64,
        );
    }
}
