//! The guest's display frames composed on this device
//! ([`crate::metal::compose`]): the compose kernel writes a scanout buffer
//! of the display the host GPU shows, from the guest RAM of the frame's
//! layers.
//!
//! The scanout buffers are the display driver's dumb buffers, which the
//! display exports as dma-bufs and this device imports once each
//! ([`NativeBackend::import_compose_target`]). A layer's planes are guest RAM
//! imported as the Metal device imports it ([`Backend::import_memory`]),
//! kept between frames while the same runs come back, and taken under
//! [`COMPOSE_TOKEN`], which no guest is told of when the import is freed.
//!
//! A frame records one command buffer of its own pool: the kernel over the
//! frame's arguments, then a write of its ticket to the compose marker,
//! which the host reads from any processor with no driver call
//! ([`NativeBackend::compose_marker`]). It goes on the Metal device's queue
//! in the service's turn, as the Metal device's own work does, and signals
//! a timeline of its own: a submission the driver refuses then keeps its
//! failure under that timeline rather than under the next Metal command
//! buffer's signal. One frame is in flight at a time; the next is refused
//! until the marker shows the last.
use super::*;
use crate::metal::compose::{Arguments, Job, Submitted, shows};

/// The token compose imports are taken under: freeing one retires nothing
/// the guest wired.
pub(super) const COMPOSE_TOKEN: NonZeroU64 = NonZeroU64::MAX;

/// Where the marker lies in the shared buffer, after the arguments.
const MARKER: u64 = 512;
/// The shared buffer's bytes: the arguments and the marker.
const SHARED_BYTES: u64 = 1024;
/// Planes kept imported at most; past it those the last frame did not use
/// are freed (into the backend's kept imports).
const KEPT_PLANES: usize = 8;

/// A scanout buffer imported from the display's driver.
struct Target {
    handle: u32,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    address: u64,
}

/// A layer plane's guest RAM imported, and the ticket of the last frame
/// that read it.
struct Plane {
    ranges: Vec<MemoryRange>,
    memory: Memory,
    buffer: Buffer,
    address: u64,
    used: u64,
}

/// What composing on this device keeps between frames.
pub(super) struct Compositor {
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    timeline: vk::Semaphore,
    /// Host-visible, coherent: the arguments, then the marker.
    shared: BufferAllocation,
    host: NonNull<u8>,
    targets: Vec<Target>,
    planes: Vec<Plane>,
    /// The ticket of the last frame submitted, zero for none.
    last: u64,
}

impl Compositor {
    fn marker(&self) -> NonNull<AtomicU32> {
        // SAFETY: the shared buffer maps `SHARED_BYTES`, the marker word
        // within them, 4-byte aligned.
        unsafe { NonNull::new_unchecked(self.host.as_ptr().add(MARKER as usize).cast()) }
    }

    /// Whether the GPU finished the last frame submitted.
    fn idle(&self) -> bool {
        // SAFETY: the marker is mapped while the compositor lives.
        let marker = unsafe { self.marker().as_ref() }.load(Ordering::Acquire);
        self.last == 0 || shows(marker, self.last)
    }
}

impl NativeBackend {
    /// The compositor, made at its first scanout buffer.
    fn compositor(&mut self) -> Result<&mut Compositor> {
        if self.compositor.is_none() {
            let made = self.make_compositor()?;
            self.compositor = Some(made);
        }
        self.compositor
            .as_mut()
            .ok_or(Error::Invalid("no compositor"))
    }

    fn make_compositor(&mut self) -> Result<Compositor> {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.queue_family);
        // SAFETY: the device is live and the family is its queue's.
        let pool = unsafe { self.device.create_command_pool(&info, None) }.map_err(failure)?;
        let made = self.compositor_parts(pool);
        if made.is_err() {
            // SAFETY: nothing allocated from the pool is in use.
            unsafe { self.device.destroy_command_pool(pool, None) };
        }
        made
    }

    fn compositor_parts(&mut self, pool: vk::CommandPool) -> Result<Compositor> {
        let commands = match CommandBuffers::allocate(&self.device, pool, 1)? {
            CommandBuffers::One(buffer) => buffer,
            CommandBuffers::Many(buffers) => buffers[0],
        };
        let timeline = handle(Backend::create_semaphore(self, 0)?.0);
        let info = BufferInfo {
            size: SHARED_BYTES,
            usage: super::super::vulkan::BufferUsage::SHADER_DEVICE_ADDRESS
                | super::super::vulkan::BufferUsage::STORAGE_BUFFER
                | super::super::vulkan::BufferUsage::TRANSFER_DST,
            flags: super::super::vulkan::BufferCreate::EMPTY,
        };
        let request = MemoryRequest {
            required: super::super::vulkan::MemoryProperties::HOST_VISIBLE
                .union(super::super::vulkan::MemoryProperties::HOST_COHERENT),
            preferred: super::super::vulkan::MemoryProperties::HOST_CACHED,
            mapped: true,
        };
        let shared = match self.create_buffer(&info, request) {
            Ok(shared) => shared,
            Err(error) => {
                // SAFETY: the semaphore was never used.
                unsafe { self.device.destroy_semaphore(timeline, None) };
                return Err(error);
            }
        };
        let Some(host) = self.host_bytes(shared.memory, 0, SHARED_BYTES) else {
            self.destroy(Object::Buffer(shared.buffer));
            self.destroy(Object::Memory(shared.memory));
            // SAFETY: as above.
            unsafe { self.device.destroy_semaphore(timeline, None) };
            return Err(Error::Unsupported(
                "coherent host memory for the compose marker",
            ));
        };
        Ok(Compositor {
            pool,
            commands,
            timeline,
            shared,
            host,
            targets: Vec::new(),
            planes: Vec::new(),
            last: 0,
        })
    }

    /// Imports the display's scanout buffer `handle`, the dma-buf `fd` of
    /// `bytes`: the driver takes the descriptor when this succeeds, and the
    /// caller keeps it otherwise.
    pub(super) fn import_compose_target(&mut self, handle: u32, fd: i32, bytes: u64) -> Result<()> {
        if self.compose == vk::Pipeline::null() {
            return Err(Error::Unsupported(
                "the display compose kernel (the driver refused it)",
            ));
        }
        self.compositor()?;
        let fns = self
            .fns
            .external_memory_fd
            .as_ref()
            .ok_or(Error::Unsupported("memory import"))?;
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: the descriptor is a dma-buf the caller holds.
        unsafe { fns.get_memory_fd_properties(DMA_BUF, fd, &mut properties) }.map_err(failure)?;
        let request = MemoryRequest {
            required: super::super::vulkan::MemoryProperties::EMPTY,
            preferred: super::super::vulkan::MemoryProperties::DEVICE_LOCAL,
            mapped: false,
        };
        let index = *objects::memory_candidates(
            &self.snapshot.memory,
            properties.memory_type_bits,
            request,
        )
        .first()
        .ok_or(Error::Unsupported("a memory type of the scanout buffer"))?;
        let info = BufferInfo {
            size: bytes,
            usage: super::super::vulkan::BufferUsage::SHADER_DEVICE_ADDRESS
                | super::super::vulkan::BufferUsage::STORAGE_BUFFER,
            flags: super::super::vulkan::BufferCreate::EMPTY,
        };
        let buffer = objects::imported_buffer(&info, DMA_BUF, |create| {
            // SAFETY: the create info outlives the call.
            unsafe { self.device.create_buffer(create, None) }.map_err(failure)
        })?;
        let mut from_fd = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(DMA_BUF)
            .fd(fd);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(bytes)
            .memory_type_index(index)
            .push_next(&mut flags)
            .push_next(&mut dedicated)
            .push_next(&mut from_fd);
        // SAFETY: the allocate info and its chain outlive the call.
        let memory = match unsafe { self.device.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(error) => {
                // SAFETY: nothing uses the new buffer.
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(failure(error));
            }
        };
        // SAFETY: the memory was imported for this buffer, whole.
        if let Err(error) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: nothing uses either.
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            return Err(failure(error));
        }
        let address = self.address(buffer, &info);
        let compositor = self.compositor()?;
        compositor.targets.push(Target {
            handle,
            memory,
            buffer,
            address,
        });
        Ok(())
    }

    /// The GPU address of the plane of `ranges`, imported now if `import`
    /// allows and the last frame no longer reads what is freed for it.
    fn compose_plane(
        &mut self,
        ranges: &[MemoryRange],
        ticket: u64,
        import: bool,
    ) -> Result<Option<u64>> {
        let compositor = self.compositor()?;
        if let Some(plane) = compositor
            .planes
            .iter_mut()
            .find(|plane| plane.ranges == ranges)
        {
            plane.used = ticket;
            return Ok(Some(plane.address));
        }
        if !import {
            return Ok(None);
        }
        let bytes = ranges.iter().map(|range| range.bytes).sum();
        let memory = self.import_memory(ranges, COMPOSE_TOKEN)?;
        let info = BufferInfo {
            size: bytes,
            usage: super::super::vulkan::BufferUsage::SHADER_DEVICE_ADDRESS
                | super::super::vulkan::BufferUsage::STORAGE_BUFFER,
            flags: super::super::vulkan::BufferCreate::EMPTY,
        };
        let placed = self.create_placed_buffer(&info, memory, 0);
        let (buffer, address) = self.or_destroy(placed, &[Object::Memory(memory)])?;
        self.compositor()?.planes.push(Plane {
            ranges: ranges.to_vec(),
            memory,
            buffer,
            address,
            used: ticket,
        });
        Ok(Some(address))
    }

    /// Frees the planes past [`KEPT_PLANES`] that the frame of `ticket`
    /// does not read; the frames before it are done.
    fn trim_planes(&mut self, ticket: u64) {
        let Some(compositor) = self.compositor.as_mut() else {
            return;
        };
        let mut freed = Vec::new();
        while compositor.planes.len() > KEPT_PLANES {
            let Some(index) = compositor
                .planes
                .iter()
                .enumerate()
                .filter(|(_, plane)| plane.used != ticket)
                .min_by_key(|(_, plane)| plane.used)
                .map(|(index, _)| index)
            else {
                break;
            };
            freed.push(compositor.planes.swap_remove(index));
        }
        for plane in freed {
            self.destroy(Object::Buffer(plane.buffer));
            self.destroy(Object::Memory(plane.memory));
        }
    }

    /// [`Backend::compose`].
    pub(super) fn compose_frame(&mut self, job: &Job<'_>) -> Result<Submitted> {
        let Some(compositor) = self.compositor.as_ref() else {
            return Ok(Submitted::NoTarget);
        };
        if !compositor.idle() {
            return Ok(Submitted::NotReady);
        }
        let Some(target) = compositor
            .targets
            .iter()
            .find(|target| target.handle == job.target)
            .map(|target| target.address)
        else {
            return Ok(Submitted::NoTarget);
        };
        let mut planes = Vec::with_capacity(job.plan.layers.len());
        for placed in &job.plan.layers {
            let layer = &placed.layer;
            let Some(color) = self.compose_plane(&layer.color.ranges, job.ticket, job.import)?
            else {
                return Ok(Submitted::NotReady);
            };
            let alpha = match &layer.alpha {
                Some(alpha) => match self.compose_plane(&alpha.ranges, job.ticket, job.import)? {
                    Some(address) => address,
                    None => return Ok(Submitted::NotReady),
                },
                None => 0,
            };
            planes.push((color, alpha));
        }
        self.trim_planes(job.ticket);
        let arguments = Arguments::encode(job.plan, target, job.pitch, &planes);
        let pipeline = self.compose;
        let layout = self.layout;
        let compositor = self.compositor()?;
        // SAFETY: the shared buffer maps `SHARED_BYTES` coherently, which
        // hold the arguments, and the GPU finished the last frame that read
        // them.
        unsafe {
            core::ptr::copy_nonoverlapping(
                arguments.as_ptr(),
                compositor.host.as_ptr(),
                arguments.len(),
            );
        }
        let (commands, timeline) = (compositor.commands, compositor.timeline);
        let (address, shared) = (
            compositor.shared.address,
            handle(compositor.shared.buffer.0),
        );
        let constants = PushConstants {
            arguments: address,
            ..PushConstants::default()
        }
        .encode();
        let device = &self.device;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // What the Metal work queued before wrote (a surface WindowServer
        // rendered) is what the kernel reads.
        let before_kernel = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        let before_marker = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER | vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::HOST_READ)];
        let after_marker = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: the command buffer is the compositor's, its last
        // submission is complete (the marker shows it), and every handle
        // recorded is live; the kernel reads only the arguments and the
        // planes they name and writes the target's rows.
        let recorded = unsafe {
            device
                .reset_command_buffer(commands, vk::CommandBufferResetFlags::empty())
                .and_then(|()| device.begin_command_buffer(commands, &begin))
                .map(|()| {
                    device.cmd_pipeline_barrier2(
                        commands,
                        &vk::DependencyInfo::default().memory_barriers(&before_kernel),
                    );
                    device.cmd_bind_pipeline(commands, vk::PipelineBindPoint::COMPUTE, pipeline);
                    device.cmd_push_constants(
                        commands,
                        layout,
                        vk::ShaderStageFlags::ALL,
                        0,
                        &constants,
                    );
                    device.cmd_dispatch(commands, Arguments::groups(job.plan), 1, 1);
                    device.cmd_pipeline_barrier2(
                        commands,
                        &vk::DependencyInfo::default().memory_barriers(&before_marker),
                    );
                    device.cmd_fill_buffer(commands, shared, MARKER, 4, job.ticket as u32);
                    device.cmd_pipeline_barrier2(
                        commands,
                        &vk::DependencyInfo::default().memory_barriers(&after_marker),
                    );
                })
                .and_then(|()| device.end_command_buffer(commands))
        };
        recorded.map_err(failure)?;
        let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(commands)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline)
            .value(job.ticket)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submits = [vk::SubmitInfo2::default()
            .command_buffer_infos(&buffers)
            .signal_semaphore_infos(&signals)];
        // SAFETY: the turn holds the queue; the arrays outlive the call.
        unsafe { device.queue_submit2(self.queue, &submits, vk::Fence::null()) }
            .map_err(failure)?;
        self.compositor()?.last = job.ticket;
        Ok(Submitted::Queued)
    }

    /// [`Backend::compose_marker`].
    pub(super) fn compose_word(&self) -> Option<NonNull<AtomicU32>> {
        self.compositor.as_ref().map(Compositor::marker)
    }

    /// Destroys the compositor's objects; the device is idle.
    pub(super) fn destroy_compositor(&mut self) {
        let Some(compositor) = self.compositor.take() else {
            return;
        };
        for plane in compositor.planes {
            self.destroy(Object::Buffer(plane.buffer));
            self.free_memory(plane.memory);
        }
        // SAFETY: the device is idle; each handle is the compositor's,
        // destroyed once.
        unsafe {
            for target in compositor.targets {
                self.device.destroy_buffer(target.buffer, None);
                self.device.free_memory(target.memory, None);
            }
            self.device.destroy_command_pool(compositor.pool, None);
            self.device.destroy_semaphore(compositor.timeline, None);
        }
        self.destroy(Object::Buffer(compositor.shared.buffer));
        self.free_memory(compositor.shared.memory);
    }
}
