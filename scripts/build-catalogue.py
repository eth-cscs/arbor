#!/usr/bin/env python3

import subprocess as sp
import sys
from tempfile import TemporaryDirectory
import os
from pathlib import Path
import shutil
import stat
import string
import argparse

def parse_arguments():
    def append_slash(s):
        return s+'/' if s and not s.endswith('/') else s

    class ConciseHelpFormatter(argparse.HelpFormatter):
        def __init__(self, **kwargs):
            super(ConciseHelpFormatter, self).__init__(max_help_position=20, **kwargs)

        def _format_action_invocation(self, action):
            if not action.option_strings:
                return super(ConciseHelpFormatter, self)._format_action_invocation(action)
            else:
                optstr = ', '.join(action.option_strings)
                if action.nargs==0:
                    return optstr
                else:
                    return optstr+' '+self._format_args(action, action.dest.upper())

    parser = argparse.ArgumentParser(
        description = 'Generate dynamic catalogue and build it into a shared object.',
        usage = '%(prog)s catalogue_name mod_source_dir',
        add_help = False,
        formatter_class = ConciseHelpFormatter)

    parser.add_argument('name',
                        metavar='name',
                        type=str,
                        help='Catalogue name.')

    parser.add_argument('modpfx',
                        metavar='modpfx',
                        type=str,
                        help='Directory name where *.mod files live.')

    parser.add_argument(
        '-h', '--help',
        action = 'help',
        help = 'display this help and exit')

    return vars(parser.parse_args())

args    = parse_arguments()
arb     = Path(__file__).parents[1]
pwd     = Path.cwd()
name    = args['name']
mod_dir = pwd / Path(args['modpfx'])
mods    = [ f[:-4] for f in os.listdir(mod_dir) if f.endswith('.mod') ]

cmake = """
cmake_minimum_required(VERSION 3.9)
project({catalogue}-cat LANGUAGES CXX)

find_package(arbor REQUIRED)

include(BuildModules.cmake)

set(ARB_WITH_EXTERNAL_MODCC true)
find_program(modcc NAMES modcc)

make_catalogue(
  NAME {catalogue}
  SOURCES "${{CMAKE_CURRENT_SOURCE_DIR}}/mod"
  OUTPUT "CAT_{catalogue}_SOURCES"
  MECHS {mods})
"""

print(f"Building catalogue '{name}' from mechanisms in {mod_dir}")
for m in mods:
    print(" *", m)

with TemporaryDirectory() as tmp:
    print(f"Setting up temporary build directory: {tmp}")
    tmp = Path(tmp)
    shutil.copytree(mod_dir, tmp / 'mod')
    os.mkdir(tmp / 'build')
    os.chdir(tmp / 'build')
    with open(tmp / 'CMakeLists.txt', 'w') as fd:
        fd.write(cmake.format(catalogue=name, mods=' '.join(mods)))
    shutil.copy2(f'{arb}/mechanisms/BuildModules.cmake', tmp)
    shutil.copy2(f'{arb}/mechanisms/generate_catalogue', tmp)
    print("Configuring...")
    sp.run('cmake .. &> /dev/null', shell=True)
    print("Building...")
    sp.run('make &> /dev/null',     shell=True)
    shutil.copy2(f'{name}.cat', pwd)
    print(f'Catalogue has been built and copied to {pwd}/{name}.cat')
