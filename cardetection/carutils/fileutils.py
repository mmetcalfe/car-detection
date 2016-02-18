import io
import os.path
import yaml

def find_in_ancestors(fname):
    """ Finds a file with the given name in the current directory its parent, or
    any ancestor.  """

    dirname = os.curdir
    while True:
        if not os.path.isdir(dirname):
            abspath = os.path.abspath(dirname)
            raise IOError('The directory \'{}\' does not exist (\'{}\').'.format(dirname, abspath))

        test_fname = os.path.join(dirname, fname)

        if os.path.isfile(test_fname):
            return test_fname

        dirname = os.path.join(dirname, os.pardir)

# loadYamlFile :: String -> IO (Tree String)
def load_yaml_file(fname):
    if not os.path.isfile(fname):
        raise ValueError('Input file \'{}\' does not exist!'.format(fname))
    file = open(fname, 'r')
    data = yaml.load(file)
    file.close()
    return data

def read_process_stdout_unbufferred(process):
    # TODO: Extract utility function:
    finished_reading = False
    while not finished_reading:
        line = []
        is_carriage_return_line = False
        while True:
            inchar = process.stdout.read(1)
            # print repr(inchar)
            # Return is the line is complete:
            if inchar == '\r' or inchar == '\n':
                is_carriage_return_line = inchar == '\r'
                break

            # If the character is not empty or None:
            if inchar:
                line.append(inchar)

            # Note: poll returns the return code if the process is finished:
            elif process.poll() is not None:
                finished_reading = True
                is_carriage_return_line = False
                break
        line = ''.join(line)
        yield line, is_carriage_return_line


# Example:
# stream = subprocess.Popen(['ls'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=output_dir, bufsize=0)
# with open('file.txt') as fh:
#    stream_to_file_observing_cr(stream, fh)
def stream_to_file_observing_cr(stream, file_obj):
    last_line_begin_pos = file_obj.tell()
    for line, is_carriage_return_line in read_process_stdout_unbufferred(stream):
        # Write the line to disk, applying carriage returns:
        # Note: This avoids saving repeated status lines, while also
        # allowing status to be viewed by opening the file, or using
        #    $ tail -f file.txt.
        if is_carriage_return_line:
            if not last_line_begin_pos:
                last_line_begin_pos = file_obj.tell()
            file_obj.seek(last_line_begin_pos)
        else:
            last_line_begin_pos = None

        file_obj.write(line)
        if is_carriage_return_line:
            # Overwrite the line:
            file_obj.write(' ' * (80-len(line)))
        file_obj.write('\n')
        file_obj.flush()
