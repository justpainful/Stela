use super::*;

#[test]
fn the_export_request_matches_linux() {
    // DRM_IOCTL_PRIME_HANDLE_TO_FD.
    assert_eq!(PRIME_HANDLE_TO_FD, 0xc00c_642d);
    assert_eq!(size_of::<PrimeHandle>(), 12);
}

/// The GPU's frame goes into the hidden buffer, which no flip shows and no
/// CPU frame fills until the GPU is done; then it is flipped as a present.
#[test]
fn a_gpu_frame_flips_only_once_finished() {
    let mut card = Card::new(vec![mode(2, 4, PREFERRED)]);
    let hidden;
    {
        let mut display = Display::take(&mut card, (2, 4)).unwrap();
        hidden = display.buffers[2].framebuffer;
        let handle = display.buffers[2].handle;
        let preview = display.gpu_target(Some(1..2));
        let target = display.begin_gpu_frame(Some(1..2)).unwrap();
        assert_eq!(preview, target);
        // A buffer that held only black needs every row.
        assert_eq!(target.rows, 0..4);
        assert_eq!(target.handle, handle);
        assert_eq!(target.pitch, display.buffers[2].pitch);
        assert!(display.gpu_writing());
        display.flush().unwrap();
        assert_eq!(display.drm.flip_requests, 0);
        assert_eq!(display.present(&[1; 32]), Err(Error::GpuWriting));
        assert!(display.prepare_rows(None, |_, _| Ok(())).is_err());
        assert_eq!(display.begin_gpu_frame(None), Err(Error::GpuWriting));
        display.finish_gpu_frame().unwrap();
        assert!(!display.gpu_writing());
        assert_eq!(display.buffers[0].framebuffer, hidden);
    }
    assert_eq!(card.flips, [hidden]);
}

/// A CPU frame queued and not yet flipped gives way to the GPU's newer one
/// in the same buffer, which already holds the older frame's other rows.
#[test]
fn a_gpu_frame_supersedes_a_queued_one() {
    let mut card = Card::new(vec![mode(2, 4, PREFERRED)]);
    let spare;
    {
        let mut display = Display::take(&mut card, (2, 4)).unwrap();
        spare = display.buffers[2].framebuffer;
        display.drm.immediate = false;
        display.present(&[1; 32]).unwrap();
        assert!(display.queued());
        let target = display.begin_gpu_frame(Some(2..3)).unwrap();
        // The queued frame filled the buffer whole: only the damage is new.
        assert_eq!(target.rows, 2..3);
        assert!(!display.queued());
        display.drm.immediate = true;
        display.flush().unwrap();
        assert_eq!(display.drm.flip_requests, 0);
        display.finish_gpu_frame().unwrap();
        assert_eq!(display.buffers[0].framebuffer, spare);
    }
    assert_eq!(card.flips, [spare]);
}

/// A GPU frame given up on shows nothing, and the buffer it was written
/// into is filled whole by the next frame.
#[test]
fn an_abandoned_gpu_frame_is_not_flipped_and_its_buffer_is_refilled() {
    let mut card = Card::new(vec![mode(2, 4, PREFERRED)]);
    {
        let mut display = Display::take(&mut card, (2, 4)).unwrap();
        // Each buffer lacks only row 1.
        for damage in [None, None, None, Some(1..2), Some(1..2), Some(1..2)] {
            display
                .prepare_rows(damage, |_, _| Ok(()))
                .unwrap()
                .present()
                .unwrap();
        }
        let flips = display.drm.flip_requests;
        assert_eq!(display.begin_gpu_frame(Some(1..2)).unwrap().rows, 1..2);
        display.abandon_gpu_frame();
        assert!(!display.gpu_writing());
        display.flush().unwrap();
        assert_eq!(display.drm.flip_requests, flips);
        let mut asked = Vec::new();
        display
            .prepare_rows(Some(3..4), |y, _| {
                asked.push(y);
                Ok(())
            })
            .unwrap()
            .present()
            .unwrap();
        assert_eq!(asked, [0, 1, 2, 3]);
    }
}

#[test]
fn only_the_displays_own_buffers_are_exported() {
    let mut card = Card::new(vec![mode(2, 2, PREFERRED)]);
    let mut display = Display::take(&mut card, (2, 2)).unwrap();
    let handle = display.buffers[2].handle;
    assert_eq!(display.export_buffer(handle), Ok(1000 + handle as i32));
    assert_eq!(display.export_buffer(7), Err(Error::InvalidBuffer));
}
