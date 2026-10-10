"""Composite mixin for OT-CFM training and sampling."""

from src.model.exp.frecas_mixin import CFMFrecasMixin
from src.model.exp.inversion_mixin import CFMInversionMixin
from src.model.cfm.loss_mixin import CFMLossMixin
from src.model.cfm.ode_mixin import CFMODEMixin


class CFMSamplingMixin(
    CFMLossMixin, CFMODEMixin, CFMFrecasMixin, CFMInversionMixin
):
    pass
