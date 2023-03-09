//! Copy of https://github.com/rust-lang/rust/blob/1.56.1/library/core/src/slice/sort.rs
//! "MODIFIED:" marks places that have been modified from the original.
//!
//! Slice sorting
//!
//! This module contains a sorting algorithm based on Orson Peters' pattern-defeating quicksort,
//! published at: <https://github.com/orlp/pdqsort>
//!
//! Unstable sorting is compatible with libcore because it doesn't allocate memory, unlike our
//! stable sorting implementation.

// ignore-tidy-undocumented-unsafe

use core::cmp;
use core::mem;
// MODIFIED: Don't use unsafe APIs from `core::ptr`.

// MODIFIED: We don't need `CopyOnDrop` (see the comment in `shift_head`/`shift_tail`).

/// Shifts the first element to the right until it encounters a greater or equal element.
fn shift_head<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // MODIFIED:
    // The standard library implementation used unsafe operations, with a `CopyOnDrop` helper
    // to keep track of the "hole" that the original first element should be copied into after
    // the loop. This was necessary because, when using direct memory copies, it would in
    // general be unsound to end up with a duplicate element in `v` in the case that `is_less`
    // panics and the panic strategy is to unwind. Instead we use `swap` to bubble the
    // original first element up to the point at which it meets a greater or equal element.

    // If v.len() < 2 then this loop is a no-op, as in the original code.
    for i in 1..v.len() {
        // v[i - 1] corresponds to tmp in the original code.
        if !is_less(&v[i], &v[i - 1]) {
            break;
        }
        v.swap(i - 1, i);
    }
}

/// Shifts the last element to the left until it encounters a smaller or equal element.
fn shift_tail<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // MODIFIED:
    // The standard library implementation used unsafe operations, with a `CopyOnDrop` helper
    // to keep track of the "hole" that the original last element should be copied into after
    // the loop. This was necessary because, when using direct memory copies, it would in
    // general be unsound to end up with a duplicate element in `v` in the case that `is_less`
    // panics and the panic strategy is to unwind. Instead we use `swap` to bubble the
    // original last element down to the point at which it meets a smaller or equal element.

    // If v.len() < 2 then this loop is a no-op, as in the original code.
    for i in (0..v.len() - 1).rev() {
        // v[i + 1] corresponds to tmp in the original code.
        if !is_less(&v[i + 1], &v[i]) {
            break;
        }
        v.swap(i + 1, i);
    }
}

/// Partially sorts a slice by shifting several out-of-order elements around.
///
/// Returns `true` if the slice is sorted at the end. This function is *O*(*n*) worst-case.
#[cold]
fn partial_insertion_sort<T, F>(v: &mut [T], is_less: &mut F) -> bool
where
    F: FnMut(&T, &T) -> bool,
{
    // Maximum number of adjacent out-of-order pairs that will get shifted.
    const MAX_STEPS: usize = 5;
    // If the slice is shorter than this, don't shift any elements.
    const SHORTEST_SHIFTING: usize = 50;

    let len = v.len();
    let mut i = 1;

    for _ in 0..MAX_STEPS {
        // Find the next pair of adjacent out-of-order elements.
        while i < len && !is_less(&v[i], &v[i - 1]) {
            i += 1;
        }

        // Are we done?
        if i == len {
            return true;
        }

        // Don't shift elements on short arrays, that has a performance cost.
        if len < SHORTEST_SHIFTING {
            return false;
        }

        // Swap the found pair of elements. This puts them in correct order.
        v.swap(i - 1, i);

        // Shift the smaller element to the left.
        shift_tail(&mut v[..i], is_less);
        // Shift the greater element to the right.
        shift_head(&mut v[i..], is_less);
    }

    // Didn't manage to sort the slice in the limited number of steps.
    false
}

/// Sorts a slice using insertion sort, which is *O*(*n*^2) worst-case.
fn insertion_sort<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    for i in 1..v.len() {
        shift_tail(&mut v[..i + 1], is_less);
    }
}

/// Sorts `v` using heapsort, which guarantees *O*(*n* \* log(*n*)) worst-case.
#[cold]
pub fn heapsort<T, F>(v: &mut [T], mut is_less: F)
where
    F: FnMut(&T, &T) -> bool,
{
    // This binary heap respects the invariant `parent >= child`.
    let mut sift_down = |v: &mut [T], mut node| {
        loop {
            // Children of `node`:
            let left = 2 * node + 1;
            let right = 2 * node + 2;

            // Choose the greater child.
            let greater = if right < v.len() && is_less(&v[left], &v[right]) {
                right
            } else {
                left
            };

            // Stop if the invariant holds at `node`.
            if greater >= v.len() || !is_less(&v[node], &v[greater]) {
                break;
            }

            // Swap `node` with the greater child, move one step down, and continue sifting.
            v.swap(node, greater);
            node = greater;
        }
    };

    // Build the heap in linear time.
    for i in (0..v.len() / 2).rev() {
        sift_down(v, i);
    }

    // Pop maximal elements from the heap.
    for i in (1..v.len()).rev() {
        v.swap(0, i);
        sift_down(&mut v[..i], 0);
    }
}

/// Partitions `v` into elements smaller than `pivot`, followed by elements greater than or equal
/// to `pivot`.
///
/// Returns the number of elements smaller than `pivot`.
///
/// Partitioning is performed block-by-block in order to minimize the cost of branching operations.
/// This idea is presented in the [BlockQuicksort][pdf] paper.
///
/// [pdf]: https://drops.dagstuhl.de/opus/volltexte/2016/6389/pdf/LIPIcs-ESA-2016-38.pdf
fn partition_in_blocks<T, F>(v: &mut [T], pivot: &T, is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    // Number of elements in a typical block.
    const BLOCK: usize = 128;

    // The partitioning algorithm repeats the following steps until completion:
    //
    // 1. Trace a block from the left side to identify elements greater than or equal to the pivot.
    // 2. Trace a block from the right side to identify elements smaller than the pivot.
    // 3. Exchange the identified elements between the left and right side.
    //
    // We keep the following variables for a block of elements:
    //
    // 1. `block` - Number of elements in the block.
    // 2. `start` - Start pointer into the `offsets` array.
    // 3. `end` - End pointer into the `offsets` array.
    // 4. `offsets - Indices of out-of-order elements within the block.
    //
    // MODIFIED: Use indices instead of pointers (and 0 instead of a null pointer).

    // The current block on the left side (`v[0..block_l]`).
    let mut l: usize = 0;
    let mut block_l = BLOCK;
    let mut start_l: usize = 0;
    let mut end_l: usize = 0;
    let mut offsets_l = [0u8; BLOCK];

    // The current block on the right side (`v[(r-block_r)..r]`).
    let mut r: usize = v.len();
    let mut block_r = BLOCK;
    let mut start_r: usize = 0;
    let mut end_r: usize = 0;
    let mut offsets_r = [0u8; BLOCK];

    // MODIFIED: Keep this even though it is now trivial, to reduce the diff from the original code.
    let width = |l: usize, r: usize| r - l;

    loop {
        // We are done with partitioning block-by-block when `l` and `r` get very close. Then we do
        // some patch-up work in order to partition the remaining elements in between.
        let is_done = width(l, r) <= 2 * BLOCK;

        if is_done {
            // Number of remaining elements (still not compared to the pivot).
            let mut rem = width(l, r);
            if start_l < end_l || start_r < end_r {
                rem -= BLOCK;
            }

            // Adjust block sizes so that the left and right block don't overlap, but get perfectly
            // aligned to cover the whole remaining gap.
            if start_l < end_l {
                block_r = rem;
            } else if start_r < end_r {
                block_l = rem;
            } else {
                block_l = rem / 2;
                block_r = rem - block_l;
            }
            debug_assert!(block_l <= BLOCK && block_r <= BLOCK);
            debug_assert!(width(l, r) == block_l + block_r);
        }

        if start_l == end_l {
            // Trace `block_l` elements from the left side.
            start_l = 0;
            end_l = 0;
            let mut elem = l;

            for i in 0..block_l {
                offsets_l[end_l] = i as u8;
                end_l += !is_less(&v[elem], pivot) as usize;
                elem += 1;
            }
        }

        if start_r == end_r {
            // Trace `block_r` elements from the right side.
            start_r = 0;
            end_r = 0;
            let mut elem = r;

            for i in 0..block_r {
                elem -= 1;
                offsets_r[end_r] = i as u8;
                end_r += is_less(&v[elem], pivot) as usize;
            }
        }

        // Number of out-of-order elements to swap between the left and right side.
        let count = cmp::min(width(start_l, end_l), width(start_r, end_r));

        if count > 0 {
            // MODIFIED: These don't need to be macros.
            let left = |sl| l + (offsets_l[sl] as usize);
            let right = |sr| r - (offsets_r[sr] as usize) - 1;

            // MODIFIED: Match the cyclic permutation performed by the original code,
            // but using swaps for safety and to avoid needing `T` to be `Copy`.
            // The efficiency motivation is lost, but we want exact equivalence.

            v.swap(left(start_l), right(start_r)); // tmp is now on the right
            for _ in 1..count {
                start_l += 1;
                v.swap(left(start_l), right(start_r)); // diagonal swap; tmp is now on the left
                start_r += 1;
                v.swap(right(start_r), left(start_l)); // tmp is now on the right
            }
            // Final copy from tmp to right in the original isn't needed; tmp is already
            // in the correct place.
            start_l += 1;
            start_r += 1;
        }

        if start_l == end_l {
            // All out-of-order elements in the left block were moved. Move to the next block.

            // block-width-guarantee
            // Correctness: If `!is_done` then the slice width is guaranteed to be at least `2*BLOCK` wide. There
            // are at most `BLOCK` elements in `offsets_l` because of its size, so the `offset` operation is
            // safe. Otherwise, the debug assertions in the `is_done` case guarantee that
            // `width(l, r) == block_l + block_r`, namely, that the block sizes have been adjusted to account
            // for the smaller number of remaining elements.
            l += block_l;
        }

        if start_r == end_r {
            // All out-of-order elements in the right block were moved. Move to the previous block.

            // Correctness: Same argument as [block-width-guarantee]. Either this is a full block `2*BLOCK`-wide,
            // or `block_r` has been adjusted for the last handful of elements.
            r -= block_r;
        }

        if is_done {
            break;
        }
    }

    // All that remains now is at most one block (either the left or the right) with out-of-order
    // elements that need to be moved. Such remaining elements can be simply shifted to the end
    // within their block.

    if start_l < end_l {
        // The left block remains.
        // Move its remaining out-of-order elements to the far right.
        debug_assert_eq!(width(l, r), block_l);
        while start_l < end_l {
            end_l -= 1;
            v.swap(l + (offsets_l[end_l] as usize), r - 1);
            r -= 1;
        }
        r
    } else if start_r < end_r {
        // The right block remains.
        // Move its remaining out-of-order elements to the far left.
        debug_assert_eq!(width(l, r), block_r);
        while start_r < end_r {
            end_r -= 1;
            v.swap(l, r - (offsets_r[end_r] as usize) - 1);
            l += 1;
        }
        l
    } else {
        // Nothing else to do, we're done.
        l
    }
}

/// Partitions `v` into elements smaller than `v[pivot]`, followed by elements greater than or
/// equal to `v[pivot]`.
///
/// Returns a tuple of:
///
/// 1. Number of elements smaller than `v[pivot]`.
/// 2. True if `v` was already partitioned.
fn partition<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> (usize, bool)
where
    F: FnMut(&T, &T) -> bool,
{
    let (mid, was_partitioned) = {
        // Place the pivot at the beginning of slice.
        v.swap(0, pivot);
        let (pivot, v) = v.split_at_mut(1);
        let pivot = &mut pivot[0];

        // MODIFIED: Don't read the pivot onto the stack; it's fine for it to reference the slice.

        // Find the first pair of out-of-order elements.
        let mut l = 0;
        let mut r = v.len();

        // Find the first element greater than or equal to the pivot.
        while l < r && is_less(&v[l], pivot) {
            l += 1;
        }

        // Find the last element smaller than the pivot.
        while l < r && !is_less(&v[r - 1], pivot) {
            r -= 1;
        }

        (
            l + partition_in_blocks(&mut v[l..r], pivot, is_less),
            l >= r,
        )
    };

    // Place the pivot between the two partitions.
    v.swap(0, mid);

    (mid, was_partitioned)
}

/// Partitions `v` into elements equal to `v[pivot]` followed by elements greater than `v[pivot]`.
///
/// Returns the number of elements equal to the pivot. It is assumed that `v` does not contain
/// elements smaller than the pivot.
fn partition_equal<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    // Place the pivot at the beginning of slice.
    v.swap(0, pivot);
    let (pivot, v) = v.split_at_mut(1);
    let pivot = &mut pivot[0];

    // MODIFIED: Don't read the pivot onto the stack; it's fine for it to reference the slice.

    // Now partition the slice.
    let mut l = 0;
    let mut r = v.len();
    loop {
        // Find the first element greater than the pivot.
        while l < r && !is_less(pivot, &v[l]) {
            l += 1;
        }

        // Find the last element equal to the pivot.
        while l < r && is_less(pivot, &v[r - 1]) {
            r -= 1;
        }

        // Are we done?
        if l >= r {
            break;
        }

        // Swap the found pair of out-of-order elements.
        r -= 1;
        v.swap(l, r);
        l += 1;
    }

    // We found `l` elements equal to the pivot. Add 1 to account for the pivot itself.
    l + 1
}

/// Scatters some elements around in an attempt to break patterns that might cause imbalanced
/// partitions in quicksort.
#[cold]
fn break_patterns<T>(v: &mut [T]) {
    let len = v.len();
    if len >= 8 {
        // Pseudorandom number generator from the "Xorshift RNGs" paper by George Marsaglia.
        let mut random = len as u32;
        let mut gen_u32 = || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random
        };
        // MODIFIED: Use two calls to `gen_u32` on all platforms, to make the algorithm deterministic.
        // It is correct to cast to usize even on 32-bit platforms, because (modulus - 1) fits in usize.
        let mut gen_usize = || (((gen_u32() as u64) << 32) | (gen_u32() as u64)) as usize;

        // Take random numbers modulo this number.
        // The number fits into `usize` because `len` is not greater than `isize::MAX`.
        let modulus = len.next_power_of_two();

        // Some pivot candidates will be in the nearby of this index. Let's randomize them.
        let pos = len / 4 * 2;

        for i in 0..3 {
            // Generate a random number modulo `len`. However, in order to avoid costly operations
            // we first take it modulo a power of two, and then decrease by `len` until it fits
            // into the range `[0, len - 1]`.
            let mut other = gen_usize() & (modulus - 1);

            // `other` is guaranteed to be less than `2 * len`.
            if other >= len {
                other -= len;
            }

            v.swap(pos - 1 + i, other);
        }
    }
}

/// Chooses a pivot in `v` and returns the index and `true` if the slice is likely already sorted.
///
/// Elements in `v` might be reordered in the process.
fn choose_pivot<T, F>(v: &mut [T], is_less: &mut F) -> (usize, bool)
where
    F: FnMut(&T, &T) -> bool,
{
    // Minimum length to choose the median-of-medians method.
    // Shorter slices use the simple median-of-three method.
    const SHORTEST_MEDIAN_OF_MEDIANS: usize = 50;
    // Maximum number of swaps that can be performed in this function.
    const MAX_SWAPS: usize = 4 * 3;

    let len = v.len();

    // Three indices near which we are going to choose a pivot.
    let mut a = len / 4 * 1;
    let mut b = len / 4 * 2;
    let mut c = len / 4 * 3;

    // Counts the total number of swaps we are about to perform while sorting indices.
    let mut swaps = 0;

    if len >= 8 {
        // Swaps indices so that `v[a] <= v[b]`.
        let mut sort2 = |a: usize, b: usize| -> (usize, usize) {
            if is_less(&v[b], &v[a]) {
                swaps += 1;
                (b, a)
            } else {
                (a, b)
            }
        };

        // Finds the median of `v[a], v[b], v[c]` and returns its index.
        // MODIFIED: Renamed from `sort3` since we only use the median.
        let mut median = |a: usize, b: usize, c: usize| -> usize {
            let (a, b) = sort2(a, b);
            let (b, _) = sort2(b, c);
            let (_, b) = sort2(a, b);
            b
        };

        if len >= SHORTEST_MEDIAN_OF_MEDIANS {
            // Find medians in the neighborhoods of `a`, `b`, and `c`.
            a = median(a - 1, a, a + 1);
            b = median(b - 1, b, b + 1);
            c = median(c - 1, c, c + 1);
        }

        // Find the median among `a`, `b`, and `c`.
        b = median(a, b, c);
    }

    if swaps < MAX_SWAPS {
        (b, swaps == 0)
    } else {
        // The maximum number of swaps was performed. Chances are the slice is descending or mostly
        // descending, so reversing will probably help sort it faster.
        v.reverse();
        (len - 1 - b, true)
    }
}

/// Sorts `v` recursively.
///
/// If the slice had a predecessor in the original array, it is specified as `pred`.
///
/// `limit` is the number of allowed imbalanced partitions before switching to `heapsort`. If zero,
/// this function will immediately switch to heapsort.
fn recurse<'a, T, F>(mut v: &'a mut [T], is_less: &mut F, mut pred: Option<&'a T>, mut limit: u32)
where
    F: FnMut(&T, &T) -> bool,
{
    // Slices of up to this length get sorted using insertion sort.
    const MAX_INSERTION: usize = 20;

    // True if the last partitioning was reasonably balanced.
    let mut was_balanced = true;
    // True if the last partitioning didn't shuffle elements (the slice was already partitioned).
    let mut was_partitioned = true;

    loop {
        let len = v.len();

        // Very short slices get sorted using insertion sort.
        if len <= MAX_INSERTION {
            insertion_sort(v, is_less);
            return;
        }

        // If too many bad pivot choices were made, simply fall back to heapsort in order to
        // guarantee `O(n * log(n))` worst-case.
        if limit == 0 {
            heapsort(v, is_less);
            return;
        }

        // If the last partitioning was imbalanced, try breaking patterns in the slice by shuffling
        // some elements around. Hopefully we'll choose a better pivot this time.
        if !was_balanced {
            break_patterns(v);
            limit -= 1;
        }

        // Choose a pivot and try guessing whether the slice is already sorted.
        let (pivot, likely_sorted) = choose_pivot(v, is_less);

        // If the last partitioning was decently balanced and didn't shuffle elements, and if pivot
        // selection predicts the slice is likely already sorted...
        if was_balanced && was_partitioned && likely_sorted {
            // Try identifying several out-of-order elements and shifting them to correct
            // positions. If the slice ends up being completely sorted, we're done.
            if partial_insertion_sort(v, is_less) {
                return;
            }
        }

        // If the chosen pivot is equal to the predecessor, then it's the smallest element in the
        // slice. Partition the slice into elements equal to and elements greater than the pivot.
        // This case is usually hit when the slice contains many duplicate elements.
        if let Some(p) = pred {
            if !is_less(p, &v[pivot]) {
                let mid = partition_equal(v, pivot, is_less);

                // Continue sorting elements greater than the pivot.
                v = &mut { v }[mid..];
                continue;
            }
        }

        // Partition the slice.
        let (mid, was_p) = partition(v, pivot, is_less);
        was_balanced = cmp::min(mid, len - mid) >= len / 8;
        was_partitioned = was_p;

        // Split the slice into `left`, `pivot`, and `right`.
        let (left, right) = { v }.split_at_mut(mid);
        let (pivot, right) = right.split_at_mut(1);
        let pivot = &pivot[0];

        // Recurse into the shorter side only in order to minimize the total number of recursive
        // calls and consume less stack space. Then just continue with the longer side (this is
        // akin to tail recursion).
        if left.len() < right.len() {
            recurse(left, is_less, pred, limit);
            v = right;
            pred = Some(pivot);
        } else {
            recurse(right, is_less, Some(pivot), limit);
            v = left;
        }
    }
}

/// Sorts `v` using pattern-defeating quicksort, which is *O*(*n* \* log(*n*)) worst-case.
pub fn quicksort<T, F>(v: &mut [T], mut is_less: F)
where
    F: FnMut(&T, &T) -> bool,
{
    // Sorting has no meaningful behavior on zero-sized types.
    if mem::size_of::<T>() == 0 {
        return;
    }

    // Limit the number of imbalanced partitions to `floor(log2(len)) + 1`.
    let limit = usize::BITS - v.len().leading_zeros();

    recurse(v, &mut is_less, None, limit);
}

// MODIFIED: Delete `partition_at_index_loop` and `partition_at_index` which we don't use.
