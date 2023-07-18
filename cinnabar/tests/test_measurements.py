import cinnabar


def test_ground():
    g1 = cinnabar.ReferenceState()
    g2 = cinnabar.ReferenceState()
    g3 = cinnabar.ReferenceState(label='MLE')
    g4 = cinnabar.ReferenceState(label='MLE')

    assert g1 == g2
    assert g1 != g3
    assert g3 == g4
