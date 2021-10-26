#!/usr/bin/env python

import os, sys, time

def files_to_timestamp(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    return dict ([(f, os.path.getmtime(f)) for f in files])


def watchFiles(path_to_watch, callback_func ):
    print('Watching {}..'.format(path_to_watch))

    before = files_to_timestamp(path_to_watch)

    while 1:
        time.sleep (2)
        after = files_to_timestamp(path_to_watch)

        added = [f for f in after.keys() if not f in before.keys()]
        removed = [f for f in before.keys() if not f in after.keys()]
        modified = []

        for f in before.keys():
            if not f in removed:
                if os.path.getmtime(f) != before.get(f):
                    modified.append(f)

        if added: 
            print('Added: {}'.format(', '.join(added)))
            callback_func()
        if removed: 
            print('Removed: {}'.format(', '.join(removed)))
            callback_func()
        if modified: 
            print('Modified: {}'.format(', '.join(modified)))
            callback_func()

        before = after


if __name__ == "__main__":
    
    path_to_watch = sys.argv[1]

    def callback():
        print("Test")

    watchFiles(path_to_watch, callback)
