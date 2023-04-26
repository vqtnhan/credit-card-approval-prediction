import os
os.chdir("./data")

if __name__ == "__main__":
    for filename in ["train", "test"]:
        with open(f"{filename}.arff", "r") as f:
            lines = f.readlines()

        for type in ["numeric", "nominal"]:
            with open(f"{filename}_{type}.arff", "w") as f:
                for line in lines:
                    if line.startswith("@attribute Target numeric"):
                        if type in line:
                            f.write(line)
                        else:
                            f.write(line.replace("numeric", "{0,1}"))
                    else:
                        f.write(line)
