"""Smoke tests for the installed MedShiftLab-CXR package."""


def test_public_package_modules_import() -> None:
    import medshiftlab
    import medshiftlab.data
    import medshiftlab.evaluation
    import medshiftlab.labels

    assert medshiftlab.__name__ == "medshiftlab"
    assert medshiftlab.data.__name__ == "medshiftlab.data"
    assert medshiftlab.evaluation.__name__ == "medshiftlab.evaluation"
    assert medshiftlab.labels.__name__ == "medshiftlab.labels"
