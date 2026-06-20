# Third-party notices and acknowledgements

`fitscube-rs` is licensed under the BSD 3-Clause License (see [LICENSE](LICENSE)).
It is a Rust port of prior work in the radio-astronomy community. This file
records the provenance of that work and reproduces the required upstream license
notices.

---

## fitscube (BSD 3-Clause)

`fitscube-rs` is a port of the **fitscube** Python package by Alec Thomson — the
image-to-cube combination, frequency/time axis construction (including even vs
uneven spacing detection), per-channel `BEAMS` table handling, bounding-box
trimming, and plane extraction all follow its design and conventions:

  https://github.com/AlecThomson/fitscube

The test suite validates `fitscube-rs` output against the upstream `fitscube`
package (installed from PyPI) to confirm parity.

fitscube is distributed under the BSD 3-Clause License:

> Copyright (c) 2023, Alec Thomson
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> [BSD 3-Clause terms — see LICENSE for the full text.]
