
# camelHumpsAcronym :: String -> String
def camel_humps_acronym(name):
    caps = filter(str.isupper, name)
    initials = name[0] + caps.lower()
    return initials

# From: http://stackoverflow.com/a/7406369
def make_filename_safe(filename):
    # Replace tuples:
    filename = filename.replace(', ', 'x')

    # Replace decimal points:
    filename = filename.replace('.', ',')

    keepcharacters = (',','_','-')
    filtered_chars = [c for c in filename if c.isalnum() or c in keepcharacters]
    safe_str = "".join(filtered_chars).rstrip()
    return safe_str
