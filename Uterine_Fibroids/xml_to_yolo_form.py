from pylabel import importer
dataset = importer.ImportVOC(path="Annotations")

data=dataset.export.ExportToYoloV5()

print(data)