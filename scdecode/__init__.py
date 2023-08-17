"""A project training models for decoding protein sidechains from CG configurations."""

# Don't import here since mostly want to run as command-line tools
# To load specific modules, can still use from scdecode import prep_pdb, data_io
# from . import prep_pdb, data_io

__all__ = [
    'prep_pdb',
    'data_io',
]
