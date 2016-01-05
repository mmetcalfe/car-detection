import argparse
import yaml
import os.path
import itertools
import copy

# type TreePath = [String]
# type ParameterSet = (TreePath, [String])
# getParameterSets :: Tree String -> [ParameterSet]
def getParameterSets(tree, path=[]):
    sets = []
    for k in tree:
        v = tree[k]
        if k == 'paramSet':
            sets = sets + [(path, tree[k])]
            # print sets
        elif isinstance(v, dict):
            # print k
            sets = sets + getParameterSets(tree[k], path + [k])
        # else:
        #     print k, tree[k]
    return sets

# applyParam :: Tree String -> [String] -> String -> ()
def applyParam(tree, path, value):
    k = path[0]

    if len(path) == 1:
        tree[k] = value
    else:
        applyParam(tree[k], path[1:], value)

# camelHumpsAcronym :: String -> String
def camelHumpsAcronym(name):
    caps = filter(str.isupper, name)
    initials = name[0] + caps.lower()
    return initials

# generateTrials :: String -> String -> IO ()
def generateTrials(template_fname, output_dir):
    # Create directories:
    if not os.path.isdir(output_dir):
    	print '\n## Creating output directory: {}'.format(output_dir)
    	os.makedirs(output_dir)
    else:
    	print '\n## Using existing output directory: {}'.format(output_dir)

    # Load template file:
    stream = open(template_fname, 'r')
    template_yaml = yaml.load(stream)
    stream.close()

    paramSets = getParameterSets(template_yaml)
    indexedParams = map(lambda (p, vs): map(lambda v: (p, v), vs), paramSets)

    # Iterate through sets of config parameters:
    for params in list(itertools.product(*indexedParams)):
        # Copy the template:
        trial = copy.deepcopy(template_yaml)

        # Overwrite configured values in the template:
        for param in params:
            p, v = param
            applyParam(trial, p, v)

        # Generate a name for the trial:
        trialName = 'trial_' + '_'.join(map(lambda (k,v): camelHumpsAcronym(k[-1]) + v, params))

        # Save the trial to the output directory:
        stream = file('{}/{}.yaml'.format(output_dir, trialName), 'w')
        yaml.dump(trial, stream)
        stream.close()

if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser(description='Generate trials')
    parser.add_argument('template_yaml', type=str, nargs='?', default='template.yaml', help='Filename of the YAML file describing the trials to generate.')
    parser.add_argument('output_dir', type=str, nargs='?', default='trials', help='Directory in which to output the generated trials.')
    args = parser.parse_args()

    generateTrials(args.template_yaml, args.output_dir)
