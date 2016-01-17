
# camel_humps_acronym :: String -> String
def camel_humps_acronym(name):
    caps = filter(str.isupper, name)
    initials = name[0] + caps.lower()
    return initials

# snake_case_acronym :: String -> String
def snake_case_acronym(name):
    non_empty_segments = filter(bool, name.split('_'))
    return ''.join(s[0] for s in non_empty_segments)

# acronym_for_name :: String -> String
def acronym_for_name(name):
    # is_camel_case = filter(str.isupper, name)
    is_snake_case = '_' in name
    if is_snake_case:
        return snake_case_acronym(name)
    else:
        return camel_humps_acronym(name)

# From: http://stackoverflow.com/a/7406369
# Make a string safe to use as a filename (or a MongoDB database name).
def make_filename_safe(filename):
    # Replace tuples:
    filename = filename.replace(', ', 'x')

    # Replace decimal points:
    filename = filename.replace('.', ',')

    keepcharacters = (',','_','-')
    filtered_chars = [c for c in filename if c.isalnum() or c in keepcharacters]
    safe_str = "".join(filtered_chars).rstrip()
    return safe_str


# safe_name_from_info_dict :: Map String object -> String -> String
# Creates a string that summarises the contents of the given dictionary.
# Used to generate different filenames for different sets of trial parameters.
def safe_name_from_info_dict(info_dict, prefix=''):
    safe_info = {}
    for k, v in info_dict.iteritems():
        # Get the safe key and value:
        safe_key = acronym_for_name(k)
        safe_val = make_filename_safe(str(v))

        # Ensure keys with the same acronym are still added:
        orig_key = safe_key
        i = 1
        while safe_key in safe_info:
            safe_key = orig_key + str(i)
            i += 1

        # Record the safe info:
        safe_info[safe_key] = safe_val

    # Must be sorted to ensure identical dictionaries return the same names:
    safe_name = prefix + '_'.join(map(lambda k: k+safe_info[k], sorted(safe_info.keys())))
    return safe_name
