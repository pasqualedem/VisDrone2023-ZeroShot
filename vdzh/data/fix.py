import os
import xmltodict

def fix_labels(root):
    # Label 26 of image 04137.jpg is [162, 329, 162, 366]
    annotations = os.path.join(root, "Annotations")
    
    
    xml_to_fix = os.path.join(annotations, "04137.xml")
    with open(xml_to_fix, "r") as f:
        annotation =  xmltodict.parse(f.read())
    # Label 26 of image 04137.jpg is [162, 329, 162, 366] (a ship)
    annotation["annotation"]["object"][26]["bndbox"]["xmax"] = 179
    with open(xml_to_fix, "w") as f:
        xmltodict.unparse(annotation, output=f)
    
    # Label 53 of image 07007.jpg is [528, 485, 529, 485] (a ship)
    xml_to_fix = os.path.join(annotations, "07007.xml")
    with open(xml_to_fix, "r") as f:
        annotation =  xmltodict.parse(f.read())
    annotation["annotation"]["object"][53]["bndbox"]["xmax"] = 545
    annotation["annotation"]["object"][53]["bndbox"]["ymax"] = 530
    with open(xml_to_fix, "w") as f:
        xmltodict.unparse(annotation, output=f)
    
    # Label 3 of image 08325.jpg is [1, 786, 1, 788] (a vehicle) (We delete it)
    xml_to_fix = os.path.join(annotations, "08325.xml")
    with open(xml_to_fix, "r") as f:
        annotation =  xmltodict.parse(f.read())
    annotation["annotation"]["object"] = \
        annotation["annotation"]["object"][:3] + annotation["annotation"]["object"][4:]
    with open(xml_to_fix, "w") as f:
        xmltodict.unparse(annotation, output=f)
        
    # Label 3 of image 06256.jpg is [632, 701, 632, 701] (a vehicle) (We delete it)
    xml_to_fix = os.path.join(annotations, "06256.xml")
    with open(xml_to_fix, "r") as f:
        annotation =  xmltodict.parse(f.read())
    annotation["annotation"]["object"] = \
        annotation["annotation"]["object"][:3] + annotation["annotation"]["object"][4:]
    with open(xml_to_fix, "w") as f:
        xmltodict.unparse(annotation, output=f)