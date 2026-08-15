from enum import IntFlag

from voussoir.tools import Capability


def test_capability_is_intflag():
    assert issubclass(Capability, IntFlag)


def test_capability_supports_flag_union():
    combined = Capability.READ_PUBLIC | Capability.READ_PRIVATE
    # Union holds both members and excludes others — assert structurally
    # so a future bit-position reshuffle doesn't silently pass.
    assert Capability.READ_PUBLIC in combined
    assert Capability.READ_PRIVATE in combined
    assert Capability.WRITE_PRIVATE not in combined
    assert Capability.EXFILTRATION not in combined
    # Numeric stability check belongs in test_capability_bit_values_preserved.
    assert int(combined) == int(Capability.READ_PUBLIC) + int(Capability.READ_PRIVATE)


def test_capability_bit_values_preserved():
    # Numeric equivalence with prior int constants
    assert int(Capability.NONE) == 0
    assert int(Capability.READ_PUBLIC) == 1
    assert int(Capability.READ_PRIVATE) == 2
    assert int(Capability.WRITE_PRIVATE) == 4
    assert int(Capability.EXFILTRATION) == 8


def test_capability_repr_is_named():
    assert "READ_PRIVATE" in repr(Capability.READ_PRIVATE)
