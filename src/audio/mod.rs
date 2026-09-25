//! Audio subsystem: the lock-free ring buffer, incoming-packet decoding into the
//! ring, and the output device / resampler. Input DSP (noise gate and gain) runs
//! client-side in the AudioWorklet, so the server is a pure passthrough on the
//! receive hot path — it only decodes bytes and hands them to the output stage.
//!
//! Submodules keep each concern — and its tests — self-contained. The public
//! surface used by the rest of the crate is re-exported here.

mod output;
mod processor;
mod ring_buffer;

#[cfg(windows)]
pub use output::rename_capture_device;
pub use output::{
    find_device, list_input_devices, list_output_devices, spawn_output_supervisor,
    suggest_discord_input, DEFAULT_DEVICE,
};
pub use processor::{decode_into_rings, MAX_SAMPLES_PER_PACKET};
pub use ring_buffer::RingBuffer;

/// Maximum sample rate accepted from a client; anything higher is ignored.
pub const MAX_SAMPLE_RATE: u32 = 192_000;
