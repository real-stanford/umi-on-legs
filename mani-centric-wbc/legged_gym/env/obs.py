from typing import Any, List, Optional

import pydantic
import torch


class Config:
    arbitrary_types_allowed = True


@pydantic.dataclasses.dataclass(config=Config)
class ObservationAttribute:
    """
    ObservationAttribute is a pydantic dataclass that defines an observation attribute
    of another dataclass. It is used to define the observation space of an IsaacGymEnv.
    Attributes:
        key: The attribute name of the dataclass.
        dim: The dimension of the observation.
        scale: The scale of the observation. If None, no scaling is applied.
        noise_std: The standard deviation of the Gaussian noise added to the observation.
            If None, no noise is added.
        clip: The clipping range of the observation. If None, no clipping is applied.
    """

    key: str
    dim: int
    scale: Optional[float]
    offset: Optional[torch.Tensor]
    noise_std: Optional[float]
    clip: Optional[float]

    def __call__(
        self, struct: Any, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        tensor = getattr(struct, self.key)
        if self.scale is not None:
            tensor = tensor * self.scale
        if self.noise_std is not None and self.noise_std > 0 and generator is not None:
            tensor = (
                tensor
                + torch.randn(
                    tensor.shape,
                    generator=generator,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                * self.noise_std
            )
        if self.offset is not None:
            tensor = tensor - self.offset
        if self.clip is not None:
            tensor = torch.clip(tensor, -self.clip, self.clip)
        return tensor
