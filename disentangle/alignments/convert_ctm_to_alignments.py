import sys

if len(sys.argv) != 3:
    print('Usage: convert_ctm_to_alignments.py <ctm> <output-alignments>')
    quit()

ctm = sys.argv[1]
op_path = sys.argv[2]

op_f = open(op_path, 'w')
current_uttid = 'init'
start_frame = 0
with open(ctm) as f:
    for line in f.readlines():
        line_uttid = line.rstrip().split()[0]

        if line_uttid != current_uttid:
            if current_uttid != 'init':
                frame_idx = int(float(tokens[2]) * 100)
                duration = int(float(tokens[3]) * 100)
                op_f.write('{} {} {}\n'.format(current_uttid, start_frame, duration))
                start_frame = frame_idx


            tokens = line.rstrip().split()
            current_uttid = tokens[0]
            start_frame = 0
            frame_idx = int(float(tokens[2]) * 100)
            start_frame = frame_idx
            if frame_idx == 0:
                continue

            # Do something
            op_f.write('{} {} {}\n'.format(current_uttid, 0, frame_idx ))

        else:
            tokens = line.rstrip().split()
            frame_idx = int(float(tokens[2]) * 100)
            duration = frame_idx - start_frame
            op_f.write('{} {} {}\n'.format(current_uttid, start_frame, duration))
            start_frame = frame_idx


frame_idx = int(float(tokens[2]) * 100)
duration = int(float(tokens[3]) * 100)
op_f.write('{} {} {}\n'.format(current_uttid, start_frame, duration))

op_f.close()
