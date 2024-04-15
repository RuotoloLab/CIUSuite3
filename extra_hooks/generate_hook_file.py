"""
This can be done on command line, but I ended up needing it a bunch so wrote out a script.
Generates a hook file given a package name for use with PyInstaller. Uses collect_submodules
to get ALL hidden imports, making it potentially overkill but seems to work well.
# author: DP
# date: 7/12/2018
"""

from PyInstaller.utils.hooks import collect_submodules

packagename = input('Enter the Package Name:')
output_file = 'hook-{}.py'.format(packagename)

hiddenimports = collect_submodules(packagename)

# format the hidden imports list as a python list in the output file
with open(output_file, 'w') as hookfile:
    hookfile.write('hiddenimports = [')
    for item in hiddenimports:
        hookfile.write('"{}",'.format(item))
    hookfile.write(']')
