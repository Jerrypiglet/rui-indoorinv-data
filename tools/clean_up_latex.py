# small script that deletes unused graphic files in a latex project.
# specify graphic directory and logfile. Unused files will be deleted in specified directory.

from pathlib import Path
import cv2

project_dir = Path('/Users/jerrypiglet/Downloads/FIPT_arxiv')
logfile = project_dir / '0-main.log'
assert logfile.exists(), 'Logfile does not exist.'
enc = 'utf-8' # For Overleaf, try 'Latin-1' if issues encountered...
log_read = open(str(logfile), encoding=enc).read()

# for filename in os.listdir(project_dir):
# for filename in project_dir.iterdir():
for filename in project_dir.rglob("*"):
    if filename.is_file and filename.suffix == '.png':
        if filename.name in log_read:
            # print(filename.name + ' in use.')
            IF_RESIZE = False
            im = cv2.imread(str(filename))
            filesize_kb = filename.stat().st_size//1024
            if im.shape[1] >= 480 and im.shape[0] >= 240 and filesize_kb > 50:
                IF_RESIZE = True
            if '/Supp/' in str(filename) or '/fail_supp/' in str(filename):
                IF_RESIZE = False
            if IF_RESIZE:
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                scale_percent = 60 # percent of original size
                H, W = float(im.shape[0]), float(im.shape[1])
                scale_percent = 100. * 320 / W
                width = int(W * scale_percent / 100.)
                height = int(H * scale_percent / 100.)
                dim = (width, height)
                im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(filename), im)
                print(filename, im.shape, filesize_kb, '-->', dim)
        else:
            filename.unlink()
            print('Removed image: ' + filename.name)
        #     if os.path.isfile(os.path.join(project_dir, filename)):
        #         print(filename + ' not in use - deleting.')
        #         os.remove(os.path.join(project_dir, filename))
        #     else:
        #         print(filename + ' is a project_dir.')