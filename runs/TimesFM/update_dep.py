from tomlkit import dump, load


def main():
    with open("./timesfm/pyproject.toml") as fp:
        pyproject = load(fp)

    pyproject["project"]["dependencies"] = [
        "einshape>=1.0.0",
        "paxml>=1.4.0",
        "praxis>=1.4.0",
        "jax[cuda12]>=0.4.26",
        "numpy>=1.26.4,<2.0.0",
        "pandas>=2.1.4",
        "scikit-learn>=1.5.1",
        "utilsforecast>=0.1.12",
    ]

    with open("./timesfm/pyproject.toml", "w") as fp:
        dump(pyproject, fp)


if __name__ == "__main__":
    main()
