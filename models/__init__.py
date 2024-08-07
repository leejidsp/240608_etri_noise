from .sample_model import SampleModel, SampleDiscriminator
from .m01_basic import M01Basic
from .m02_ecnn import M02ECNN
from .hifigan_discriminator import MultiScaleDiscriminator, MultiPeriodDiscriminator

__all__ = [
            M01Basic,
            M02ECNN,

            SampleModel,
            SampleDiscriminator,

            MultiScaleDiscriminator,
            MultiPeriodDiscriminator,
            ]

all_model_dict ={
        'M01Basic': M01Basic,
        'M02ECNN': M02ECNN,

        'SampleModel':	SampleModel,
        'SampleDiscriminator': SampleDiscriminator,
        
        'MultiScaleDiscriminator':  MultiScaleDiscriminator,
        'MultiPeriodDiscriminator': MultiPeriodDiscriminator,
        }

