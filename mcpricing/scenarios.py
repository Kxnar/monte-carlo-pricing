"""Prespecified synthetic scenarios, not fitted market distributions."""
from .model import Mixture

SCENARIOS = {
    "gaussian": Mixture(weights=(.8,.2),locations=(0.,0.),scales=(.2*2**.5,.2*2**.5),shapes=(2.,2.)),
    "mixture": Mixture(),
    "stress": Mixture(weights=(.7,.3),locations=(.025,-.15),scales=(.14,.36),shapes=(2.,1.3)),
}
