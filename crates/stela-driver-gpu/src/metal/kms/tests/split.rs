//! Two displays on one node: the guest's main display and its second, each
//! on its own output, controller and buffers.
use super::outputs::{at, two_monitors};
use super::*;
use core::cell::RefCell;

/// One card two displays share, as both share the GPU's one DRM node.
struct Shared<'a>(&'a RefCell<Card>);

// SAFETY: the card's own contract; each request borrows it for the request
// alone.
unsafe impl Drm for Shared<'_> {
    fn control(&mut self, request: u32, argument: *mut c_void) -> i64 {
        let mut card = self.0.borrow_mut();
        let mut card: &mut Card = &mut card;
        card.control(request, argument)
    }

    fn map(&mut self, offset: u64, bytes: usize) -> Option<NonNull<u8>> {
        let mut card = self.0.borrow_mut();
        let mut card: &mut Card = &mut card;
        card.map(offset, bytes)
    }

    fn unmap(&mut self, pixels: NonNull<u8>, bytes: usize) {
        let mut card = self.0.borrow_mut();
        let mut card: &mut Card = &mut card;
        card.unmap(pixels, bytes);
    }

    fn immediate(&self) -> bool {
        self.0.borrow().immediate
    }
}

/// The main display as the take gives it, and its second's own mode.
fn split(card: &RefCell<Card>) -> Split<Shared<'_>> {
    Display::take_split(
        Shared(card),
        Shared(card),
        Wanted::Native((4, 2)),
        Wanted::Native((0, 0)),
        &|_, _, _| true,
    )
    .unwrap()
}

/// The output the take gives the picture to keeps it, and in its own size
/// now that the other no longer mirrors it; the other output runs its own
/// monitor's mode, on the other controller, from buffers of its own.
#[test]
fn each_display_has_its_own_output_controller_mode_and_buffers() {
    let card = RefCell::new(two_monitors());
    let (main_front, second_front);
    {
        let Split { main, second } = split(&card);
        let second = second.unwrap();
        assert_eq!((main.size(), main.refresh_millihertz()), ((8, 4), 60_000));
        assert_eq!(
            (second.size(), second.refresh_millihertz()),
            ((6, 4), 60_000)
        );
        assert_eq!((main.outputs(), main.unshown()), (1, 0));
        assert_eq!((second.outputs(), second.unshown()), (1, 0));
        assert_eq!((main.heads[0].connector, main.heads[0].crtc), (50, 40));
        assert_eq!((second.heads[0].connector, second.heads[0].crtc), (51, 41));
        // Each holds the other's output apart.
        assert_eq!(main.held.crtcs, [41]);
        assert_eq!(second.held.crtcs, [40]);
        assert!(!main.held.takes(51) && main.held.takes(50));
        assert!(!second.held.takes(50) && second.held.takes(51));
        let framebuffers = |display: &Display<Shared<'_>>| {
            display
                .buffers
                .iter()
                .map(|buffer| buffer.framebuffer)
                .collect::<Vec<_>>()
        };
        let mine = framebuffers(&main);
        assert!(framebuffers(&second).iter().all(|id| !mine.contains(id)));
        main_front = main.buffers[0].framebuffer;
        second_front = second.buffers[0].framebuffer;
        // Each connector is named once, by the display that takes it.
        assert_eq!(main.listings.len() + second.listings.len(), 2);
    }
    let card = card.into_inner();
    assert_eq!(card.set_40, Some((at(8, 4, PREFERRED, 60), main_front)));
    assert_eq!(card.set, Some((at(6, 4, PREFERRED, 60), second_front)));
    // Six buffers: three each.
    assert_eq!(card.memory.len(), 6);
}

/// Each display flips its own frames on its own controller, and a flip
/// the engine is busy with on one stays queued there, holding nothing of
/// the other's.
#[test]
fn a_busy_flip_on_one_display_does_not_hold_the_other() {
    let card = RefCell::new(two_monitors());
    // Controller 41, the second display's, is busy for two requests.
    card.borrow_mut().busy = 2;
    let Split { mut main, second } = split(&card);
    let mut second = second.unwrap();
    let (main_spare, second_spare) = (main.buffers[2].framebuffer, second.buffers[2].framebuffer);
    second.present(&[2; 96]).unwrap();
    assert!(second.queued());
    main.present(&[1; 128]).unwrap();
    assert!(!main.queued());
    assert_eq!(card.borrow().flips_40, [main_spare]);
    // The main display goes on while the second's flip waits.
    let main_back = main.buffers[2].framebuffer;
    main.present(&[3; 128]).unwrap();
    main.flush().unwrap();
    assert!(!main.queued());
    assert_eq!(card.borrow().flips_40, [main_spare, main_back]);
    assert!(card.borrow().flips.is_empty());
    // The second's flushes take its flip once its engine is free; the
    // main's controller sees none of them.
    second.flush().unwrap();
    assert!(second.queued());
    second.flush().unwrap();
    assert!(!second.queued());
    assert_eq!(card.borrow().flips, [second_spare]);
    assert_eq!(card.borrow().flips_40, [main_spare, main_back]);
}

/// Each display reads the vertical blanks of its own controller.
#[test]
fn each_display_reads_its_own_controllers_vertical_blank() {
    let card = RefCell::new(two_monitors());
    card.borrow_mut().vblank = Some((1, 1000));
    card.borrow_mut().vblank_41 = Some((1, 2000));
    let Split { mut main, second } = split(&card);
    let mut second = second.unwrap();
    assert_eq!(main.last_vblank_ns(), Some(1000));
    assert_eq!(second.last_vblank_ns(), Some(2000));
    assert_eq!(card.borrow().sequences, [40, 41]);
    // One display asleep reads none; the other still does.
    second.set_power(false).unwrap();
    assert_eq!(second.last_vblank_ns(), None);
    assert_eq!(main.last_vblank_ns(), Some(1000));
    assert!(card.borrow().set_40.is_some() && card.borrow().set.is_none());
}

/// The second display's monitor unplugged and plugged again is set again
/// by the second display alone: the main display never takes its output,
/// and the main display's own monitor coming back leaves the second's
/// controller alone.
#[test]
fn a_hot_plug_on_one_display_does_not_cross_to_the_other() {
    let card = RefCell::new(two_monitors());
    let Split { mut main, second } = split(&card);
    let mut second = second.unwrap();
    let second_front = second.buffers[0].framebuffer;
    let lost = Watched {
        lost: 1,
        ..Watched::default()
    };
    let restored = Watched {
        restored: 1,
        ..Watched::default()
    };
    card.borrow_mut().connected = false;
    assert_eq!(main.watch(), Watched::default());
    assert_eq!(second.watch(), lost);
    card.borrow_mut().connected = true;
    // The main display does not join it, however often it looks.
    for _ in 0..=RESTORE_TRIES {
        assert_eq!(main.watch(), Watched::default());
    }
    assert_eq!(main.outputs(), 1);
    let sets = card.borrow().mode_sets;
    assert_eq!(second.watch(), restored);
    // Off, then set again in its own mode on its own buffer.
    assert_eq!(card.borrow().mode_sets, sets + 2);
    assert_eq!(
        card.borrow().set,
        Some((at(6, 4, PREFERRED, 60), second_front))
    );

    // The main display's monitor goes and returns: controller 41 untouched.
    let main_front = main.buffers[0].framebuffer;
    card.borrow_mut().first_connected = false;
    assert_eq!(second.watch(), Watched::default());
    assert_eq!(main.watch(), lost);
    card.borrow_mut().first_connected = true;
    let shown = card.borrow().set;
    assert_eq!(second.watch(), Watched::default());
    assert_eq!(main.watch(), restored);
    assert_eq!(card.borrow().set, shown);
    assert_eq!(
        card.borrow().set_40,
        Some((at(8, 4, PREFERRED, 60), main_front))
    );
}

/// With one usable output there is no second display, and the main one
/// is the take's without a split.
#[test]
fn one_output_gives_the_take_without_a_split() {
    let card = RefCell::new(Card::new(vec![at(6, 4, PREFERRED, 60), at(4, 2, 0, 60)]));
    let Split { main, second } = split(&card);
    assert_eq!(second.err(), Some(Error::NoOutput));
    assert_eq!(main.size(), (6, 4));
    assert_eq!(main.held, Held::default());
    let mut alone = Card::new(vec![at(6, 4, PREFERRED, 60), at(4, 2, 0, 60)]);
    let taken = Display::take_fitting(&mut alone, Wanted::Native((4, 2)), &|_, _, _| true).unwrap();
    assert_eq!(
        (taken.size(), taken.outputs(), taken.heads[0].crtc),
        (main.size(), main.outputs(), main.heads[0].crtc)
    );
}

/// The driver refuses the second display's mode set: there is no second
/// display, and its output is the main display's again to try at a look.
#[test]
fn a_refused_second_display_leaves_its_output_to_the_main_one() {
    let card = RefCell::new(two_monitors());
    card.borrow_mut().refused_crtc = Some(41);
    let Split { main, second } = split(&card);
    assert_eq!(second.err(), Some(Error::Request(-22)));
    assert_eq!(main.held, Held::default());
    assert_eq!((main.outputs(), main.unshown()), (1, 1));
    assert_eq!(main.size(), (8, 4));
    // Every buffer the refused take made went back.
    assert_eq!(card.borrow().destroyed.len(), 3);
}

/// The connected outputs are counted as last detected, with no probe.
#[test]
fn the_connected_outputs_are_counted_without_a_probe() {
    let mut card = two_monitors();
    assert_eq!(connected_outputs(&mut &mut card), Ok(2));
    card.first_connected = false;
    assert_eq!(connected_outputs(&mut &mut card), Ok(1));
    assert_eq!(card.probes, 0);
}

/// A controller or encoder another display holds is never free.
#[test]
fn what_another_display_holds_is_never_free() {
    let held = Held {
        crtcs: vec![40],
        encoders: vec![60],
        ..Held::default()
    };
    assert_eq!(free_crtc(0b11, &[40, 41], &[], &held), Some(41));
    assert_eq!(free_crtc(0b01, &[40, 41], &[], &held), None);
    assert_eq!(free_encoder(&[60, 61], &[], &held), Some(61));
    assert_eq!(free_encoder(&[60], &[], &held), None);
    assert_eq!(free_crtc(0b01, &[40, 41], &[], &Held::default()), Some(40));
}
