from pathlib import Path

errors_path = Path("error_file.txt")
file = open(errors_path, "r")
content=file.readlines()
file.close()
LEs = 0
Fzs = 0
missingParts = 0
otherReasons = []
for i in range(int(len(content)/2)):
    data_path = content[2*i]
    reason = content[2*i+1]
    if "fz not in" in reason:
        Fzs += 1
    elif "montage: LE" in reason:
        LEs += 1
    elif "contiguous second" in reason:
        missingParts += 1
    else:
        otherReasons.append(reason)
total = 7361
totalmissing = LEs+Fzs+missingParts+len(otherReasons)
print("LE montage instead of Referential montage:", LEs, LEs/total)
print("Fz electrode missing:", Fzs, Fzs/total)
print("There is a missing part:", missingParts, missingParts/total)
print("Other reasons:", len(otherReasons), len(otherReasons)/total)
print("Total errors:", totalmissing, totalmissing/total)