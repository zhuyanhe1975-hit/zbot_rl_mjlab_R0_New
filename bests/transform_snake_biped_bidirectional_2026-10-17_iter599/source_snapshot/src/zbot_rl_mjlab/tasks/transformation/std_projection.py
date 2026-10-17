"""Keep bounded Gaussian exploration parameters inside the differentiable interval."""

import math

import torch
from rsl_rl.modules import GaussianDistribution
from torch.utils.hooks import RemovableHandle


class ActionStdProjection:
    """Project parameters in place without replacing them or optimizer state.

    Bounds themselves have a zero clamp derivative in the installed torch build.
    An optimizer post hook therefore keeps parameters strictly inside the declared
    range before the next policy distribution is constructed.
    """

    def __init__(self, distribution):
        self.parameter = None
        if not isinstance(distribution, GaussianDistribution):
            return
        self.is_log = distribution.std_type == "log"
        if self.is_log:
            self.parameter = distribution.log_std_param
            self.bounds = tuple(distribution.log_std_range)
        else:
            self.parameter = distribution.std_param
            self.bounds = tuple(distribution.std_range)
        # Fail at installation, rather than midway through an optimizer update.
        self._projection_bounds = self._interior_bounds()

    def _interior_bounds(self):
        parameter = self.parameter
        if parameter is None:
            return None
        low, high = self.bounds
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Action std bounds must be finite and strictly ordered")
        lower = parameter.new_tensor(low)
        upper = parameter.new_tensor(high)
        inner_lower = torch.nextafter(lower, upper)
        inner_upper = torch.nextafter(upper, lower)
        if self.is_log:
            # nextafter(0, +/-inf) is subnormal and CUDA may flush it back to 0.
            # Use an ordinary epsilon when possible, with a normal lower bound.
            info = torch.finfo(parameter.dtype)
            inset = max(info.tiny, min(info.eps, (high - low) / 4))
            if low == 0:
                inner_lower = parameter.new_tensor(inset)
            if high == 0:
                inner_upper = parameter.new_tensor(-inset)
        valid = (
            (lower < inner_lower) & (inner_lower <= inner_upper) & (inner_upper < upper)
        )
        if not valid.item():
            raise ValueError(
                "Action std range has no representable strict interior "
                f"for {parameter.dtype}: {self.bounds}"
            )
        return lower, upper, inner_lower, inner_upper

    @torch.no_grad()
    def project(self) -> None:
        if self.parameter is None:
            return
        bounds = self._projection_bounds
        if bounds is None:
            return
        lower, upper, inner_lower, inner_upper = bounds
        self.parameter.copy_(
            torch.where(
                self.parameter <= lower,
                inner_lower,
                torch.where(self.parameter >= upper, inner_upper, self.parameter),
            )
        )

    def install(self, optimizer: torch.optim.Optimizer) -> RemovableHandle:
        """Register one native post-step hook; caller owns handle lifetime."""

        def project_after_step(optimizer, args, kwargs):
            self.project()

        return optimizer.register_step_post_hook(project_after_step)
