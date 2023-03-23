import cinnabar


def test_ground():
    g1 = cinnabar.GroundState()
    g2 = cinnabar.GroundState()
    g3 = cinnabar.GroundState(label='MLE')
    g4 = cinnabar.GroundState(label='MLE')

    assert g1 == g2
    assert g1 != g3
    assert g3 == g4
