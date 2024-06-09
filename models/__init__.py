from .sample_model import SampleModel, SampleDiscriminator
from .m01_basic import M01Basic
from .hifigan_discriminator import MultiScaleDiscriminator, MultiPeriodDiscriminator

__all__ = [
            M01Basic,

            SampleModel,
            SampleDiscriminator,

            MultiScaleDiscriminator,
            MultiPeriodDiscriminator,
            ]

all_model_dict ={
        'M01Basic': M01Basic,

        'SampleModel':	SampleModel,
        'SampleDiscriminator': SampleDiscriminator,
        
        'MultiScaleDiscriminator':  MultiScaleDiscriminator,
        'MultiPeriodDiscriminator': MultiPeriodDiscriminator,
        }

