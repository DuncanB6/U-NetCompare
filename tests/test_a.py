from comp_unet.Functions import schedule


def test_schedule():
    assert schedule(1, 2) == 2
