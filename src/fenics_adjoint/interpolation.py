import dolfin
from pyadjoint.overloaded_type import create_overloaded_object


def interpolate(*args, **kwargs):
    """Interpolation is overloaded to ensure that the returned Function object is overloaded.
    We are not able to annotate the interpolation call at the moment.

    """
    kwargs.pop("ad_block_tag", None)
    output = dolfin.interpolate(*args, **kwargs)
    return create_overloaded_object(output)
