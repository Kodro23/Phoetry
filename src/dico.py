"""
Only a dico with labels associated to a key
"""
import json

class Label_dico():
    """
    A simple class derived from dict() to deal with labels
    """
    dico : dict
    """
    A dico with :
    key --> theme
    list --> labels
    """

    def __init__(self):
        """
        Initialize the dico by charging the file ./data/dico.json
        """
        self.dico = dict()

        with open('./data/dico.json', 'r') as json_file:
            self.dico = json.load(json_file)


    def add_label(self, key : str, labels = [str]):
        """
        Used to add a theme (key) and its associated labels (list of strings)
        Return nothing
        """
        if key in self.dico.keys():
            print(f"this key -{key}- is already used\n"
            "you must give another key")
        else:    
            self.dico[key] = labels
            print(f"-{key}- and its labels have been added to dico\n"
            "In order to record the dico you should used the '.export_dico()' method")

    def export(self):
        
        with open('./data/dico.json', 'w') as json_file:
            json.dump(self.dico, json_file)
            
        print("The dico has been well exported")

