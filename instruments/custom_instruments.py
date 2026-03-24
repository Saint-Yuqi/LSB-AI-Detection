"""
Reusable telescope / CCD / filter component definitions and instrument factories.

All geometric and optical parameters are fixed here. Runtime parameters
(distance, LOS, rsp_opts, paths) live in config files, NOT in this module.
"""
from __future__ import annotations


def build_arrakihs_vis2_legacy():
    """Factory: ARRAKIHS VIS2 legacy instrument.

    Geometry: 1072x1072 px, 24 µm pixel size, iSIM-170 @ 1500 mm focal.
    Filter:   BPASS230_ARK_VIS2.
    """
    from astropy import units as u
    from pNbody.Mockimgs import ccd, filters, instrument, telescope

    return instrument.Instrument(
        name="arrakihs_vis",
        telescope=telescope.Telescope(name="iSIM-170", focal=1500 * u.mm),
        ccd=ccd.CCD(
            name="arrakihs_vis_legacy",
            shape=[1072, 1072],
            pixel_size=[24 * u.micron, 24 * u.micron],
        ),
        filter_type=filters.Filter("BPASS230_ARK_VIS2"),
    )
