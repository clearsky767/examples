import time
import os

def read_txt(filename):
    fo = open(filename, "r")
    fl = fo.read()
    fo.close()
    return fl

def write_txt(filename,data):
    fo = open(filename, "w+")
    fl = fo.write(data)
    fo.close()

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return

def get_jsonlist(path):
    jsonlist = []
    for f in os.listdir(path):
        if os.path.splitext(f)[1] == ".json":
            jsonlist.append(f)
    return jsonlist

def main():
    print("start!")
    jsonlist = get_jsonlist(".")
    print(jsonlist)
    content = "{"
    for json in jsonlist:
        c = read_txt(json)
        content += c[1:-1]
        content += ","
    print("end!")
    content = content[0:-1]
    content += "}"
    write_txt("out.json", content)

if __name__ == '__main__':
    main()