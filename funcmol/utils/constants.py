PADDING_INDEX = 999

ELEMENTS_HASH = {
    "C": 0,
    "H": 1,
    "O": 2,
    "N": 3,
    "F": 4,
    "S": 5,
    "Cl": 6,
    "Br": 7,
    "P": 8,
    "I": 9,
    "B": 10,
}

# For now, using tha atomic radii from https://github.com/gnina/libmolgrid/blob/master/src/atom_typer.cpp
# which is the same used in AutoDock v4.
radiusSingleAtom = {
    "MOL": {
        "C": 2.0,
        "H": 1.0,
        "O": 1.6,
        "N": 1.75,
        "F": 1.545,
        "S": 2.0,
        "Cl": 2.045,
        "Br": 2.165,
        "P": 2.1,
        "I": 2.36,
        "B": 2.04,
    }
}
