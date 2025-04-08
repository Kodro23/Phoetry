"""
This part of code enable only to manage the dico of labels.
"""

from src.dico import Label_dico

label_dico = Label_dico()
print(label_dico.dico.keys())

key = 'nourriture'
labels = [
        "pomme",
        "banane",
        "carotte",
        "tomate",
        "poulet",
        "poisson",
        "riz",
        "pâtes",
        "fromage",
        "lait",
        "œuf",
        "pain",
        "huile",
        "sel",
        "poivre",
        "sucre",
        "farine",
        "chocolat",
        "miel",
        "yaourt",
        "épinard",
        "brocoli",
        "avocat",
        "citron",
        "ail",
        "oignon",
        "persil",
        "basilic",
        "thym",
        "romarin"
    ]

label_dico.add_label(key = key, labels = labels)
print(label_dico.dico.keys())
label_dico.export()