a = {
    '1':'a',
    '2':'b',
    '3':'c'
}

b = {
    '2':'b',
    '3':'c'
}

c = {
    '3':'c'
}

print(a.__contains__(b))
print(b in a)
print(a in c)
print(c in a)
print(b in c)
print(c in b)