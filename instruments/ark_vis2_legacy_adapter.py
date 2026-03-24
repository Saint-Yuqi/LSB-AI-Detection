"""
Thin adapter for pNbody CLI ``--instrument`` file mode.

``mockimgs_sb_compute_images`` does ``exec(open(opt.instrument).read(), globals())``
and expects a top-level ``instrument`` object.
"""
from instruments.custom_instruments import build_arrakihs_vis2_legacy

instrument = build_arrakihs_vis2_legacy()
