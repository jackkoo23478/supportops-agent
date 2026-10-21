from supportops.cli import main


def test_package_imports():
    import supportops

    assert supportops is not None


def test_cli_prints_ready(capsys):
    main()
    out = capsys.readouterr().out
    assert "ready" in out
