# ny-gpu

`ny-gpu` contains GPU-backed bound propagation support and benchmark helpers.
The crate is optional from the perspective of algorithm design, but included in
the default CLI build when the platform supports `wgpu`.

Keep device setup, shader dispatch, and GPU benchmark support inside this crate.
Higher-level verification policy belongs in `ny-propagate` or `ny-cli`.
